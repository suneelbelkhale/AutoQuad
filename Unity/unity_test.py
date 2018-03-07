from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="drone_sim_v1", worker_id=0)

print(env)

print("Success!")