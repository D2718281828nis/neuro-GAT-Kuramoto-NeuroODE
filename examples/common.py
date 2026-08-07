import torch
from torch import nn
from hmb_kuramoto_ode.utils.seed import seed_everything
from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph,batch_edges
from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE

def supervised_demo(kind="message_passing"):
    seed_everything(7); x=torch.randn(24,4); y=(x[:,0]>0).long(); model=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,2)); opt=torch.optim.Adam(model.parameters(),.05)
    initial=float(nn.functional.cross_entropy(model(x),y))
    for _ in range(25): opt.zero_grad(); loss=nn.functional.cross_entropy(model(x),y); loss.backward(); opt.step()
    acc=float((model(x).argmax(1)==y).float().mean()); print(f"{kind}: initial_loss={initial:.4f} final_loss={float(loss):.4f} accuracy={acc:.3f}")

def diffusion_demo():
    seed_everything(7); x=torch.randn(6,3); initial=float(x.var())
    for _ in range(10): x=.5*x+.25*(torch.roll(x,1,0)+torch.roll(x,-1,0))
    print(f"GRAND diffusion: variance {initial:.4f} -> {float(x.var()):.4f}")

def kuramoto_demo():
    seed_everything(7); phase=torch.randn(8); before=float(abs(torch.exp(1j*phase).mean()))
    for _ in range(60): phase=phase+.05*torch.sin(phase[None,:]-phase[:,None]).mean(1)
    print(f"Kuramoto: order_parameter {before:.4f} -> {float(abs(torch.exp(1j*phase).mean())):.4f}")

def ode_demo():
    from hmb_kuramoto_ode.models.ode_solver import integrate
    rate=nn.Parameter(torch.tensor(-1.)); y=integrate(lambda t,z:rate*z,torch.ones(3),t1=1.,step_size=.1); y.sum().backward()
    print(f"Neural ODE: y(1)={float(y[0]):.4f} rate_gradient={float(rate.grad):.4f}")

def multitask_demo():
    seed_everything(7); b,r=2,3; base,types=build_hierarchical_graph(r); edge=batch_edges(base,r*5,b); x=torch.randn(b,r,5,6); pairs=edge[:,:12]
    model=HierarchicalKuramotoODE(edge,hidden=8); opt=torch.optim.Adam(model.parameters(),.01); target=torch.tensor([0,1]); losses=[]
    for _ in range(4):
        opt.zero_grad(); out=model(x,pairs); loss=nn.functional.cross_entropy(out["graph_logits"],target)+.1*out["node_prediction"].square().mean()+.1*nn.functional.binary_cross_entropy_with_logits(out["link_logits"],torch.ones(12)); loss.backward(); opt.step(); losses.append(float(loss))
    print(f"synthetic multitask: loss {losses[0]:.4f} -> {losses[-1]:.4f}; graph_accuracy={float((out['graph_logits'].argmax(1)==target).float().mean()):.3f}; nfe={out['nfe']}")
