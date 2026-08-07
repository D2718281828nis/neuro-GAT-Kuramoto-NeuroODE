import numpy as np
import pytest
import torch
from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph,batch_edges,CROSS_FREQUENCY_LOCAL
from hmb_kuramoto_ode.data.connectivity import sample_negative_edges
from hmb_kuramoto_ode.data.preprocessing import RhythmPreprocessor,TrainNormalizer
from hmb_kuramoto_ode.data.splits import grouped_folds,assert_disjoint
from hmb_kuramoto_ode.models.attention import EdgeAttention
from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE
from hmb_kuramoto_ode.models.ode_solver import integrate

def test_preprocessing_shape_and_train_statistics():
    raw=np.random.default_rng(1).normal(size=(14,512)); p=RhythmPreprocessor(); f=p.transform_window(raw)
    assert f.shape==(14,5,6) and np.isfinite(f).all()
    n=TrainNormalizer().fit(f[None],["train"]); assert n.fitted_subjects=={"train"}; assert np.isfinite(n.transform(f[None])).all()

def test_graph_types_batching_and_five_bands():
    edge,typ=build_hierarchical_graph(14); assert edge.max()<70 and (typ==CROSS_FREQUENCY_LOCAL).any()
    bat=batch_edges(edge,70,2); assert not (((bat[0]<70)&(bat[1]>=70))|((bat[1]<70)&(bat[0]>=70))).any()

def test_attention_neighbor_normalization():
    edge,_=build_hierarchical_graph(3); h=torch.randn(15,8); a=EdgeAttention(8)(h,edge)
    for node in edge[1].unique(): assert torch.allclose(a[edge[1]==node].sum(),torch.tensor(1.),atol=1e-6)

def test_rk4_gradient_and_finite_state():
    rate=torch.nn.Parameter(torch.tensor(.2)); out=integrate(lambda t,y:rate*y,torch.ones(2),t1=.2,step_size=.05); out.sum().backward()
    assert torch.isfinite(out).all() and rate.grad is not None and torch.isfinite(rate.grad)

def test_multitask_shapes_all_gradients_and_coupling_bounds(tmp_path):
    edge,_=build_hierarchical_graph(3); edge=batch_edges(edge,15,2); x=torch.randn(2,3,5,6); pairs=edge[:,:10]
    model=HierarchicalKuramotoODE(edge,hidden=8); out=model(x,pairs,torch.ones(2,3,dtype=torch.bool)); loss=out["graph_logits"].sum()+out["node_prediction"].sum()+out["link_logits"].sum(); loss.backward()
    assert out["graph_logits"].shape==(2,2) and out["node_prediction"].shape==(2,3,5,6) and out["link_logits"].shape==(10,)
    assert 0 <= model.field.coupling.min() and model.field.coupling.max() <= 2
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    path=tmp_path/"m.pt"; torch.save(model.state_dict(),path); clone=HierarchicalKuramotoODE(edge,hidden=8); clone.load_state_dict(torch.load(path,weights_only=True)); model.eval(); clone.eval()
    assert torch.allclose(model(x,pairs)["graph_logits"],clone(x,pairs)["graph_logits"])

def test_pooling_mask_and_negative_edges():
    edge,_=build_hierarchical_graph(2); model=HierarchicalKuramotoODE(edge,hidden=8); out=model(torch.randn(1,2,5,6),channel_mask=torch.tensor([[True,False]]))
    assert out["attention"]["region"][0,1]==0
    pos=torch.tensor([[0,1,2],[1,2,3]]); neg=sample_negative_edges(5,pos,4,torch.Generator().manual_seed(1)); known={tuple(sorted(x)) for x in pos.T.tolist()}; assert not known & {tuple(x) for x in neg.T.tolist()}

def test_subject_grouping_and_intentional_random_split_guard():
    subjects=np.array(["a","a","b","b","c","c"]); labels=np.array([0,1,0,1,0,1])
    for tr,te in grouped_folds(labels,subjects,3): assert_disjoint(subjects[tr],subjects[te])
    with pytest.raises(ValueError,match="leakage"): assert_disjoint(subjects[:3],subjects[2:])

def test_reproducible_forward():
    edge,_=build_hierarchical_graph(2); x=torch.randn(1,2,5,6)
    torch.manual_seed(4); a=HierarchicalKuramotoODE(edge,hidden=8); torch.manual_seed(4); b=HierarchicalKuramotoODE(edge,hidden=8); a.eval(); b.eval()
    assert torch.equal(a(x)["graph_logits"],b(x)["graph_logits"])
