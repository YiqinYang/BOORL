Dear Reviewers:

	Thanks for your efforts and time in reviewing our paper. This is the implementation of Bayesian Offline-to-Online Reinforcement Learning : A Realist Approach.
	
	First, please install the corresponding dependencies: 

	1. [Mujoco 1.50](http://www.mujoco.org/).
	2. [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py).
	
	3. [OpenAI gym 0.17.0](https://github.com/openai/gym). 

	4. [D4RL datasets](https://github.com/rail-berkeley/d4rl).
	
	5. [PyTorch 1.4.0](https://github.com/pytorch/pytorch).

	6. Python 3.6.

	For the offline training, please run: 

```
python3 main.py --save_model --env hopper-medium-v2
```

	For the online finetuning, please run:

```
python3 finetune.py --load_model --offdataset --env hopper-medium-v2
```
	
	Thanks again for your comments, and looking forward to the discussion.
	
	Best wishes,
	
The authors