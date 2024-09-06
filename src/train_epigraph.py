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
 
def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad
   
def train(model, args, log=True):
     
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
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
    pbar = tqdm(range(starting_step, 100))

    data_sampler_args = {}
    task_sampler_args = {}

    task = task_sampler(**task_sampler_args)
 
    z0 = 10
    z1 = 15 
    loss_func = task.get_training_metric()
    count = 0
    for i in pbar:
        
        model.train() 
        for t in range(100):
            for k in range(5): 
                optimizer.zero_grad() 
                xs = data_sampler.sample_xs(
                    curriculum.n_points,
                    bsize,
                    curriculum.n_dims_truncated,
                    **data_sampler_args,
                )
                task = task_sampler(**task_sampler_args)
                ys, layer_activations = task.evaluate(xs)
                
                layer_activations = [act.to(device) for act in layer_activations]
                xs = xs.to(device)
                ys = ys.to(device)
                f0, f1, f2 = model(xs, ys, loss_func, layer_activations=layer_activations)
                if model.opt_algo == 'epigraph':
                    f0_acc = (f0 + f1 + f2) 
                    f1_acc = (f1 + f2) 
                    f = [f0_acc-z1-z0, f1_acc-z1, f2-4.5] 
                else:
                    f = [f0-z1-z0, f1-z1, f2-4.5] 
                idx = torch.argmax(torch.tensor(f))  
                f[idx].backward()   
                
                optimizer.step()
                if count % args.wandb.log_every_steps == 0 and log: 
                    wandb.log(
                        {
                            "Hyperparam/z0": z0,
                            "Hyperparam/z1": z1, 
                            'Hyperparam/layer_id': idx, 
                            "Train/step 3 (y)": f0, 
                            "Train/step 2 (s^2)": f1,
                            "Train/step 1 (s^1)": f2, 
                            "n_points": curriculum.n_points,
                            "n_dims": curriculum.n_dims_truncated,
                        },
                        step=count,
                    )
                count += 1
                curriculum.update()
            z1 = z1 - 0.005
            z1 = max(z1, args.training.multilayer.z_lb)
       
        z0 = z0 - 0.2
        z0 = max(z0,args.training.multilayer.z_lb) 
         
          
        pbar.set_description(f"loss {f0}")
        if count % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and count % args.training.keep_every_steps == 0
            and not args.test_run
            and count > 0
        ) or (count == args.training.train_steps - 1):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{count}.pt"))


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
