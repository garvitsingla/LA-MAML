import gym

class EnvFactory:
    """
    Pickle-safe callable to create a Gym environment.
    """
    __slots__ = ("env_name", "env_kwargs")
    def __init__(self, env_name, env_kwargs=None):
        self.env_name = env_name
        # ensure it's a plain dict, never None
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
    def __call__(self):
        import gym
        return gym.make(self.env_name, **self.env_kwargs)

# Then replace make_env with:
def make_env(env_name, env_kwargs=None):
    return EnvFactory(env_name, env_kwargs)

# def _make_env(env_name, env_kwargs):
#     return lambda: gym.make(env_name, **(env_kwargs or {}))

# def make_env(env_name, env_kwargs=None):
#     return _make_env(env_name, env_kwargs or {})


# def make_env(env_name, env_kwargs={}, seed=None):
#     def _make_env():
#         env = gym.make(env_name, **env_kwargs)
#         if hasattr(env, 'seed'):
#             env.seed(seed)
#         return env
#     return _make_env

class Sampler(object):
    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 policy,
                 seed=None,
                 env=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.seed = seed

        if env is None:
            env = gym.make(env_name, **env_kwargs)
        self.env = env
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        return self.sample_async(*args, **kwargs)
