import os
import shutil
from random import randint
import uuid
import datetime
import numpy as np 

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
import random  
from eval import build_evals


import wandb

torch.backends.cudnn.benchmark = True
 
 
def vanilla(model, xs, ys, optimizer, loss_func, layer_activations=None, z0=0, z1=0, optimizer_theta1=None, optimizer_theta2=None, epsilon=10.):
     
    f0, f1, f2 = model(xs, ys, loss_func, layer_activations=layer_activations)
    f0.backward()
    optimizer.step() 
    idx = 0
    return f0, f1, f2, idx, None

def penalty(model, xs, ys, optimizer, loss_func, layer_activations=None, z0=0, z1=0, optimizer_theta1=None, optimizer_theta2=None, epsilon=10.):
     
    f0, f1, f2 = model(xs, ys, loss_func, layer_activations=layer_activations)
    idx = 0
    ((f0 + f1 + f2)/3).backward()
    optimizer.step() 
    return f0, f1, f2, idx, None
 
def train_step(model, xs, ys, optimizer, loss_func, layer_activations=None, z0=0, z1=0, optimizer_theta1=None, optimizer_theta2=None, epsilon=10.):
    optimizer.zero_grad()
    f0, f1, f2, idx, extra = globals()[model.opt_algo](model, xs, ys, optimizer, loss_func, layer_activations, z0, z1, optimizer_theta1, optimizer_theta2, epsilon)
    return f0.detach().item(), f1.detach().item(), f2.detach().item(), idx, extra 
 
     
def train(model, args, log=True):
     
    if model.opt_algo in ['switching_gradient', 'switching_gradient_single', 'epigraph_cot']:
        optimizer = torch.optim.Adam(model.model.parameters(), lr=args.training.learning_rate)
        optimizer1 = torch.optim.Adam(model.model1.parameters(), lr=args.training.learning_rate*2)
        optimizer2 = torch.optim.Adam(model.model2.parameters(), lr=args.training.learning_rate*2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
        optimizer1 = None 
        optimizer2 = None
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    data_sampler_args = {}
    task_sampler_args = {}

    task = task_sampler(**task_sampler_args)
 
    
    loss_func = task.get_training_metric()
    for i in pbar: 
        model.train()
        # additional inner loop to update z
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        ) 
         
        task = task_sampler(**task_sampler_args)
        ys, layer_activations = task.evaluate(xs)
        layer_activations = [act.to(device) for act in layer_activations]
         
        f0, f1, f2, layer_id, extra = train_step(model, xs.to(device), ys.to(device), optimizer,
                        loss_func, layer_activations=layer_activations, z0=z0, z1=z1, optimizer_theta1=optimizer1, optimizer_theta2=optimizer2, epsilon=curriculum.epsilon)
        if model.opt_algo[:8] == 'epigraph':
            grad_norm1, grad_norm2, grad_norm3 = extra

        if i % args.wandb.log_every_steps == 0 and log:
            
            if model.opt_algo == 'epigraph':
                wandb.log(
                    {
                        "Grad norm/layer 0": grad_norm1, 
                        "Grad norm/layer 1": grad_norm2,
                        "Grad norm/layer 2": grad_norm3,
                    }
                )
            wandb.log(
                { 
                    'Hyperparam/layer_id': layer_id, 
                    "Train/step 3 (y)": f0, 
                    "Train/step 2 (s^2)": f1,
                    "Train/step 1 (s^1)": f2, 
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated
                },
                step=i,
            )
             
        curriculum.update()

        pbar.set_description(f"loss {f0}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ) or (i == args.training.train_steps - 1):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args, log=True):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
        log = False
    else:
        if log:
            wandb.init(
                dir=args.out_dir,
                project=args.wandb.project,
                entity=args.wandb.entity,
                config=args.__dict__,
                notes=args.wandb.notes,
                name=args.wandb.name,
                mode="disabled" if args.debug_mode else "online",
                resume=True,
            )
        else:
            pass

    model = build_model(args.model)
    model.to(device)
    model.train()

    train(model, args, log)

    if args.debug_mode:
        # delete wandb directory when done
        print("Deleting out_dir {} because of debug mode".format(args.out_dir))
        shutil.rmtree("{}".format(args.out_dir), ignore_errors=True)


if __name__ == "__main__":
    device = torch.device('cuda')
    log = True 
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2_nn"]
    print(f"Running with: {args}") 
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if args.debug_mode:
        args.out_dir = "../models/debug"

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            import datetime
            now = datetime.datetime.now()
            run_id = str(now) # str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
        # add a timestamp here
        args.wandb['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args, log)
