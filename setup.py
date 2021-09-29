from setuptools import setup

setup(
   name='berl',
   version='0.1.0',
   author='Paul Templier',
   author_email='paul.templier@isae-supaero.fr',
   packages=['berl'],
   license='MIT',
   description='Benchmarking Evolutionnary Reinforcement Learning',
   long_description=open('README.md').read(),
   install_requires=[
       "torch", "gym", "matplotlib", "numpy", "dopamine-rl", "stable-baselines3",
       "stable-baselines3[extra]", "tqdm", "wandb", "cma"
   ],
)

