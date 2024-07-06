from setuptools import setup, find_packages

with open('README_PYPI.md') as f:
    long_description = f.read()

setup(
    name='jaxagents',
    url='https://github.com/amavrits/jax-agents',
    author='Antonis Mavritsakis',
    author_email='amavrits@gmail.com',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "jaxlib==0.4.28",
        "jax==0.4.28",
        "chex==0.1.86",
        "distrax==0.1.5",
        "flax==0.8.4",
        "flashbax==0.1.2",
        "jaxtyping==0.2.29",
        "gymnax==0.0.6",
        "jax-tqdm==0.2.1",
        "optax==0.2.2",
        "orbax-checkpoint==0.5.14",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "matplotlib==3.9.0",
        "pytest==8.2.1",
        ],
    extras_require={
        "dev": []
    },
    python_requires=">=3.9.0",
    version='0.1.3',
    license='MIT',
    description='Implementation of Reinforcement Learning agents in JAX',
)