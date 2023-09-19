import importlib
import random
import time
#time.sleep(10)
from collections import defaultdict
from datetime import datetime
from itertools import count
from pathlib import Path
import sys
import joblib
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import psutil
from ReadWriteMemory import ReadWriteMemory
import trackmania_rl.agents.iqn as iqn
from trackmania_rl import buffer_management, misc, nn_utilities
from trackmania_rl.buffer_utilities import buffer_collate_function
from trackmania_rl.experience_replay.basic_experience_replay import ReplayBuffer_async
import win32process
import win32gui
#from trackmania_rl.experience_replay.basic_experience_replay import ReplayBuffer
base_dir = Path(__file__).resolve().parents[1]

run_name = "TEST12"
map_name = "map5"
def start_TMI():
    workdir="C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever"
    (commandLine , processAttributes , threadAttributes , bInheritHandles , dwCreationFlags , newEnvironment , currentDirectory , startupinfo) = (None, None,None,0,0,None,workdir,win32process.STARTUPINFO())   
    (Process,Thread,dwProcessId, dwThreadId) = win32process.CreateProcess("C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever\TMInterface.exe", commandLine , processAttributes , threadAttributes , bInheritHandles , dwCreationFlags , newEnvironment , currentDirectory , startupinfo )
    return Process
def remove_fps_cap():
    # from @Kim on TrackMania Tool Assisted Discord server
    process = filter(lambda pr: pr.name() == "TmForever.exe", psutil.process_iter())
    rwm = ReadWriteMemory()

    for p in process:
        pid = int(p.pid)
        print(pid)
        process = rwm.get_process_by_id(pid)
        process.open()
        process.write(0x005292F1, 4294919657)
        process.write(0x005292F1 + 4, 2425393407)
        process.write(0x005292F1 + 8, 2425393296)
        process.close()
        print(f"Disabled FPS cap of process {pid}")
        
def CollectData(accumulated_stats,interface_name,base_dir,zone_centers,buffer_test_Queue,buffer_Queue,tensorBoard_Queue, save_dir,buffer_Lock,tensorBoard_Lock):
    from trackmania_rl import tm_interface_manager
    if misc.write_worker_prints_to_file:
        PrintOutput = open(save_dir/(interface_name+'.txt'), 'a')
        sys.stdout = PrintOutput
    print(interface_name,flush=True)
    model1 = torch.jit.script(
        iqn.Agent(
            float_inputs_dim=misc.float_input_dim,
            float_hidden_dim=misc.float_hidden_dim,
            conv_head_output_dim=misc.conv_head_output_dim,
            dense_hidden_dimension=misc.dense_hidden_dimension,
            iqn_embedding_dimension=misc.iqn_embedding_dimension,
            n_actions=len(misc.inputs),
            float_inputs_mean=misc.float_inputs_mean,
            float_inputs_std=misc.float_inputs_std,
        )
    ).to("cuda", memory_format=torch.channels_last)
    trainer = iqn.ExplorationTrainer(
        model=model1,
        iqn_k=misc.iqn_k,       
        epsilon=misc.epsilon,
        epsilon_boltzmann=misc.epsilon_boltzmann,
        tau_epsilon_boltzmann=misc.tau_epsilon_boltzmann,
        tau_greedy_boltzmann=misc.tau_greedy_boltzmann,
    )
    tmi = tm_interface_manager.TMInterfaceManager(
        base_dir=base_dir,
        running_speed=misc.running_speed,
        run_steps_per_action=misc.tm_engine_step_per_action,
        max_overall_duration_ms=misc.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=misc.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        interface_name=interface_name,
        zone_centers=zone_centers,
        
    )
    

    
    
    
    for loop_number in count(1):
        while True:
            try:
                model1.load_state_dict(torch.load(save_dir / "weights1.torch"))
                break
            except:
                print("FILE CORRUPTED RETRYING", flush=True)
                pass
        # ===============================================
        #   PLAY ONE ROUND
        # ===============================================
        rollout_start_time = time.time()
    
        is_explo = (loop_number % misc.explo_races_per_eval_race) > 0
        tmi.run_steps_per_action
        if is_explo:
            tmi.run_steps_per_action=misc.tm_engine_step_per_action
            if misc.use_random_exponential_epsilon:
                epsilon = np.random.exponential(misc.high_exploration_ratio * misc.epsilon)
                epsilon_boltzmann =  np.random.exponential(misc.high_exploration_ratio * misc.epsilon_boltzmann)                
            else:
                epsilon = misc.high_exploration_ratio * misc.epsilon
                epsilon_boltzmann =  (misc.high_exploration_ratio * misc.epsilon_boltzmann)    
                
                
           
            trainer.epsilon = (
                np.random.exponential(misc.high_exploration_ratio * epsilon)
                if accumulated_stats["cumul_number_memories_generated"] < misc.number_memories_generated_high_exploration_early_training//misc.num_sessions
                else epsilon
            )
            trainer.epsilon_boltzmann = (
                np.random.exponential(misc.high_exploration_ratio * epsilon_boltzmann)
                if accumulated_stats["cumul_number_memories_generated"] < misc.number_memories_generated_high_exploration_early_training//misc.num_sessions
                else epsilon_boltzmann
            )
        else:
            tmi.run_steps_per_action=misc.tm_engine_step_per_action_eval
            trainer.epsilon = 0
            trainer.epsilon_boltzmann = 0
            print("EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL")
        #print("Before Race Start", flush=True)
        rollout_results, end_race_stats, sucess = tmi.rollout(
            exploration_policy=trainer.get_exploration_action,
            is_eval=not is_explo,
        )
        #print("RACE DONE, SUCESS=", sucess, flush=True)
        if not sucess:
            tmi = tm_interface_manager.TMInterfaceManager(
                base_dir=base_dir,
                running_speed=misc.running_speed,
                run_steps_per_action=misc.tm_engine_step_per_action,
                max_overall_duration_ms=misc.cutoff_rollout_if_race_not_finished_within_duration_ms,
                max_minirace_duration_ms=misc.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
                interface_name=interface_name,
                zone_centers=zone_centers,
                
            )
            continue
    
        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])  ## these only count for this session not all sessions now
    
       # ===============================================
       #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
       # ===============================================
        race_stats_to_write = {
           "race_time_ratio": end_race_stats["race_time_for_ratio"] / ((time.time() - rollout_start_time) * 1000),
           "explo_race_time" if is_explo else "eval_race_time": end_race_stats["race_time"] / 1000,
           "explo_race_finished" if is_explo else "eval_race_finished": end_race_stats["race_finished"],
           "mean_action_gap": -(
               np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
           ).mean(),
           "single_zone_reached": len(rollout_results["zone_entrance_time_ms"]) - 1,
           "time_to_answer_normal_step": end_race_stats["time_to_answer_normal_step"],
           "time_to_answer_action_step": end_race_stats["time_to_answer_action_step"],
           "time_between_normal_on_run_steps": end_race_stats["time_between_normal_on_run_steps"],
           "time_between_action_on_run_steps": end_race_stats["time_between_action_on_run_steps"],
           "time_to_grab_frame": end_race_stats["time_to_grab_frame"],
           "time_between_grab_frame": end_race_stats["time_between_grab_frame"],
           "time_A_rgb2gray": end_race_stats["time_A_rgb2gray"],
           "time_A_geometry": end_race_stats["time_A_geometry"],
           "time_A_stack": end_race_stats["time_A_stack"],
           "time_exploration_policy": end_race_stats["time_exploration_policy"],
           "time_to_iface_set_set": end_race_stats["time_to_iface_set_set"],
           "time_after_iface_set_set": end_race_stats["time_after_iface_set_set"],
       }
        print("Race time ratio  ", race_stats_to_write["race_time_ratio"])
    
        if end_race_stats["race_finished"]:
            race_stats_to_write["explo_race_time_finished" if is_explo else "eval_race_time_finished"] = end_race_stats["race_time"] / 1000
        maxraceq=-9999999
        for i in range(len(misc.inputs)):
            if end_race_stats[f"q_value_{i}_starting_frame"]>maxraceq:
                maxraceq=end_race_stats[f"q_value_{i}_starting_frame"]
        for i in range(len(misc.inputs)):
            race_stats_to_write[f"q_value_{i}_starting_frame"] = end_race_stats[f"q_value_{i}_starting_frame"]#-maxraceq
    
        with tensorBoard_Lock:
            tensorBoard_Queue.put(race_stats_to_write)
        #walltime_tb = float(accumulated_stats["cumul_training_hours"] * 3600) + time.time() - time_last_save
        #for tag, value in race_stats_to_write.items():
        #    tensorboard_writer.add_scalar(
        #        tag=tag,
        #        scalar_value=value,
        #        global_step=accumulated_stats["cumul_number_frames_played"],
        #        walltime=walltime_tb,
        #   )
    
        # ===============================================
        #   SAVE STUFF IF THIS WAS A GOOD RACE
        # ===============================================
        
        print("MEean Q Gap:")
        print(-(
                np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
            ).mean(),flush=True)

        print("Expected Gap was: ",misc.expected_Q_value_difference_good_action_bad_action,flush=True)
        
        if end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"]:
            # This is a new alltime_minimum
    
            accumulated_stats["alltime_min_ms"] = end_race_stats["race_time"]
    
            sub_folder_name = f"{end_race_stats['race_time']}"
            (save_dir / "best_runs" / sub_folder_name).mkdir(parents=True, exist_ok=True)
            joblib.dump(
                rollout_results["actions"],
                save_dir / "best_runs" / sub_folder_name / f"actions.joblib",
            )
            joblib.dump(
                rollout_results["q_values"],
                save_dir / "best_runs" / sub_folder_name / f"q_values.joblib",
            )
            torch.save(
                model1.state_dict(),
                save_dir / "best_runs" / sub_folder_name / "weights1.torch",
            )
            #torch.save(
            #    model2.state_dict(),
            #    save_dir / "best_runs" / sub_folder_name / "weights2.torch",   
            #)
            #torch.save(
            #    optimizer1.state_dict(),
            #    save_dir / "best_runs" / sub_folder_name / "optimizer1.torch",
            #)
        if end_race_stats["race_time"] < misc.good_time_save_all_ms:
            sub_folder_name = f"{end_race_stats['race_time']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            (save_dir / "good_runs" / sub_folder_name).mkdir(parents=True, exist_ok=True)
            joblib.dump(
                rollout_results["actions"],
                save_dir / "good_runs" / sub_folder_name / f"actions.joblib",
            )
            joblib.dump(
                rollout_results["q_values"],
                save_dir / "good_runs" / sub_folder_name / f"q_values.joblib",
            )
            torch.save(
                model1.state_dict(),
                save_dir / "good_runs" / sub_folder_name / "weights1.torch",
            )
            #torch.save(
            #    model3.state_dict(),
            #    save_dir / "good_runs" / sub_folder_name / "weights2.torch",
            #)
            #torch.save(
            #    optimizer1.state_dict(),
            #    save_dir / "good_runs" / sub_folder_name / "optimizer1.torch",
            #)
    
        # ===============================================
        #   FILL BUFFER WITH (S, A, R, S') transitions
        # ===============================================
        if (is_explo or misc.use_eval_runs_in_buffer) and sucess:
            
            number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule_async(
                buffer_Queue,
                buffer_test_Queue,
                rollout_results,
                misc.n_steps,
                misc.gamma,
                misc.discard_non_greedy_actions_in_nsteps,
                misc.n_zone_centers_in_inputs,
                zone_centers,
                buffer_Lock
            )
    
        accumulated_stats["cumul_number_memories_generated"] += number_memories_added
        importlib.reload(misc)
        #accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
        #    misc.number_times_single_memory_is_used_before_discard * number_memories_added
        #)
        #print(f" NMG={accumulated_stats['cumul_number_memories_generated']:<8}")


if __name__ == '__main__':
    zone_centers = np.load(str(base_dir / "maps" / "map.npy"))

    # ========================================================
    # ARTIFICIALLY ADD MORE ZONE CENTERS AFTER THE FINISH LINE
    # ========================================================
    for i in range(misc.n_zone_centers_in_inputs):
        zone_centers = np.vstack(
            (
                zone_centers,
                (2 * zone_centers[-1] - zone_centers[-2])[None, :],
            )
        )

    save_dir = base_dir / "save" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=str(base_dir / "tensorboard" / run_name))

    layout = {
        run_name: {
            "eval_race_time_finished": [
                "Multiline",
                [
                    "eval_race_time_finished",
                ],
            ],
            "explo_race_time_finished": [
                "Multiline",
                [
                    "explo_race_time_finished",
                ],
            ],
            "loss": ["Multiline", ["loss$", "loss_test$"]],
            "values_starting_frame": [
                "Multiline",
                [f"q_value_{i}_starting_frame" for i in range(len(misc.inputs))],
            ],
            "single_zone_reached": [
                "Multiline",
                [
                    "single_zone_reached",
                ],
            ],
            r"races_finished": ["Multiline", ["explo_race_finished", "eval_race_finished"]],
            "iqn_std": [
                "Multiline",
                [f"std_within_iqn_quantiles_for_action{i}" for i in range(len(misc.inputs))],
            ],
            "race_time_ratio": ["Multiline", ["race_time_ratio"]],
            "mean_action_gap": [
                "Multiline",
                [
                    "mean_action_gap",
                ],
            ],
            "layer_L2": [
                "Multiline",
                [
                    "layer_.*_L2",
                ],
            ],
            "lr_ratio_L2": [
                "Multiline",
                [
                    "lr_ratio_.*_L2",
                ],
            ],
            "exp_avg_L2": [
                "Multiline",
                [
                    "exp_avg_.*_L2",
                ],
            ],
            "exp_avg_sq_L2": [
                "Multiline",
                [
                    "exp_avg_sq_.*_L2",
                ],
            ],
            "eval_race_time": [
                "Multiline",
                [
                    "eval_race_time$",
                ],
            ],
            "explo_race_time": [
                "Multiline",
                [
                    "explo_race_time$",
                ],
            ],
        },
    }
    tensorboard_writer.add_custom_scalars(layout)

    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    random_seed = 444
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # ========================================================
    # Create new stuff
    # ========================================================
    model1 = torch.jit.script(
        iqn.Agent(
            float_inputs_dim=misc.float_input_dim,
            float_hidden_dim=misc.float_hidden_dim,
            conv_head_output_dim=misc.conv_head_output_dim,
            dense_hidden_dimension=misc.dense_hidden_dimension,
            iqn_embedding_dimension=misc.iqn_embedding_dimension,
            n_actions=len(misc.inputs),
            float_inputs_mean=misc.float_inputs_mean,
            float_inputs_std=misc.float_inputs_std,
        )
    ).to("cuda", memory_format=torch.channels_last)
    model2 = torch.jit.script(
        iqn.Agent(
            float_inputs_dim=misc.float_input_dim,
            float_hidden_dim=misc.float_hidden_dim,
            conv_head_output_dim=misc.conv_head_output_dim,
            dense_hidden_dimension=misc.dense_hidden_dimension,
            iqn_embedding_dimension=misc.iqn_embedding_dimension,
            n_actions=len(misc.inputs),
            float_inputs_mean=misc.float_inputs_mean,
            float_inputs_std=misc.float_inputs_std,
        )
    ).to("cuda", memory_format=torch.channels_last)
    print(model1)

    optimizer1 = torch.optim.RAdam(model1.parameters(), lr=misc.learning_rate, eps=1e-6)
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=misc.learning_rate, eps=0.01)
    # optimizer1 = torch.optim.SGD(model1.parameters(), lr=misc.learning_rate, momentum=0.8)
    scaler = torch.cuda.amp.GradScaler()

    accumulated_stats = defaultdict(int)
    # ========================================================
    # Load existing stuff
    # ========================================================
    # noinspection PyBroadException
    try:
        model1.load_state_dict(torch.load(save_dir / "weights1.torch"))
        model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
        optimizer1.load_state_dict(torch.load(save_dir / "optimizer1.torch"))
        print(" =========================     Weights loaded !     ================================")
    except:
        print(" Could not load weights")
    torch.save(model1.state_dict(), save_dir / "weights1.torch")
    torch.save(model2.state_dict(), save_dir / "weights2.torch")
    torch.save(optimizer1.state_dict(), save_dir / "optimizer1.torch")
    # noinspection PyBroadException
    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        print(" =========================      Stats loaded !      ================================")    
    except:
        print(" Could not load stats")

    if accumulated_stats["alltime_min_ms"] == 0:
        accumulated_stats["alltime_min_ms"] = 99999999999
    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]

    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []

    # ========================================================
    # Make the trainer
    # ========================================================
    trainer = iqn.Trainer(
        model=model1,
        model2=model2,
        optimizer=optimizer1,
        scaler=scaler,
        batch_size=misc.batch_size,
        iqn_k=misc.iqn_k,
        iqn_n=misc.iqn_n,
        iqn_kappa=misc.iqn_kappa,
        epsilon=misc.epsilon,
        epsilon_boltzmann=misc.epsilon_boltzmann,
        gamma=misc.gamma,
        AL_alpha=misc.AL_alpha,
        tau_epsilon_boltzmann=misc.tau_epsilon_boltzmann,
        tau_greedy_boltzmann=misc.tau_greedy_boltzmann,
    )

    # ========================================================
    # Training loop
    # ========================================================
    model1.train()
    time_last_save = time.time()
    #pinned_buffer_Lock=mp.Lock()
    

    #pinned_buffer_size=(misc.memory_size_per_session + 10000) *misc.num_sessions
    #pinned_buffer = torch.empty((pinned_buffer_size, 1, misc.H_downsized, misc.W_downsized), dtype=torch.uint8)
    #torch.cuda.cudart().cudaHostRegister(
    #pinned_buffer.data_ptr(), pinned_buffer_size * misc.H_downsized * misc.W_downsized, 0
    #)
    #pinned_buffer=buffer_management.Pinned_buffer_async(pinned_buffer_size,pinned_buffer_Lock)

    #pinned_buffer_Queue = mp.Queue()
    #pinned_buffer.addReciever(pinned_buffer_Queue)
    
    tensorBoard_Queue=mp.Queue()
    tensorBoard_Lock=mp.Lock()
    
    remove_fps_cap()
    buffer_Lock=mp.Lock()
    buffer = ReplayBuffer_async(capacity=int(misc.memory_size/misc.num_sessions* (1-misc.buffer_test_ratio)), batch_size=misc.batch_size, collate_fn=buffer_collate_function,buffer_Lock=buffer_Lock,accumulated_stats=accumulated_stats)
    buffer_test = ReplayBuffer_async(
        capacity=int(misc.memory_size/misc.num_sessions * misc.buffer_test_ratio), batch_size=misc.batch_size, collate_fn=buffer_collate_function,buffer_Lock=buffer_Lock
    )  
    buffer_Queue = mp.Queue()
    buffer_test_Queue = mp.Queue()

    buffer.addReciever(buffer_Queue)
    buffer_test.addReciever(buffer_test_Queue)
    #Buffers=[]
    #Buffers_test=[]
    #BufferLocks=[]
    #BufferQueues=[]
    #BufferQueues_test=[]
    processes= []
    for i in range(misc.lowest_tm_interface,misc.lowest_tm_interface+misc.num_sessions):

        
        #Buffers.append(buffer)
        #Buffers_test.append(buffer_test)
        #BufferLocks.append(buffer_Lock)
        #BufferQueues.append(buffer_Queue)
        #BufferQueues_test.append(buffer_test_Queue)
        
        p=mp.Process(target=CollectData,args=(accumulated_stats,"TMInterface"+str(i),base_dir,zone_centers,buffer_test_Queue,buffer_Queue,tensorBoard_Queue,save_dir,buffer_Lock,tensorBoard_Lock))   
        p.start()        # ===============================================
        processes.append(p)
        #p2=mp.Process(target=CollectData,args=(accumulated_stats,"TMInterface1",base_dir,pinned_buffer_Queue,zone_centers,buffer_test_Queue,buffer_Queue,save_dir))   
        #p2.start()              #   LEARN ON BATCH
        #p3=mp.Process(target=CollectData,args=(accumulated_stats,"TMInterface2",base_dir,pinned_buffer_Queue,zone_centers,buffer_test_Queue,buffer_Queue,save_dir))   
        #p3.start()  
        #p4=mp.Process(target=CollectData,args=(accumulated_stats,"TMInterface3",base_dir,pinned_buffer_Queue,zone_centers,buffer_test_Queue,buffer_Queue,save_dir))   
        #p4.start()             # ===============================================
    for loop_number in count(1):
        print(loop_number)
        
        time.sleep(misc.sleep_time_between_training)
        for i in range(0,tensorBoard_Queue.qsize()):
            try:
                with tensorBoard_Lock:
                    race_stats_to_write=tensorBoard_Queue.get_nowait()
                walltime_tb = float(accumulated_stats["cumul_training_hours"] * 3600) + time.time() - time_last_save
                for tag, value in race_stats_to_write.items():
                    tensorboard_writer.add_scalar(
                        tag=tag,
                        scalar_value=value,
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                   )
            except:
                pass
        #for i in range(0,1000):
        #pinned_buffer.CheckRecievers()
        #for buffer in Buffers:
        buffer.CheckRecievers()
        print("the buffer has" , len(buffer), "items")
        #for buffer_test in Buffers_test:
        buffer_test.CheckRecievers()
        
        
        for i in range(0,misc.batches_per_training):
            #buffer = random.choice(Buffers)
            #buffer_test = random.choice(Buffers_test)
            if (
                
                len(buffer) >= misc.memory_size_start_learn
                
            ):
                if (random.random() < misc.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    loss = trainer.train_on_batch( buffer_test, do_learn=False)
                    loss_test_history.append(loss)
                    print(f"BT   {loss=:<8.2e}")
                else:
                    train_start_time = time.time()
                    loss = trainer.train_on_batch( buffer, do_learn=True)
                    accumulated_stats["cumul_number_single_memories_used"] += misc.batch_size
                    train_on_batch_duration_history.append(time.time() - train_start_time)
                    loss_history.append(loss)
                    accumulated_stats["cumul_number_batches_done"] += 1
                    print(f"B    {loss=:<8.2e}")
        
                    nn_utilities.custom_weight_decay(model1, 1 - misc.weight_decay)
        
                    # ===============================================
                    #   UPDATE TARGET NETWORK
                    # ===============================================
                    if (
                        accumulated_stats["cumul_number_single_memories_used"]
                        >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
                    ):
                        accumulated_stats["cumul_number_target_network_updates"] += 1
                        accumulated_stats[
                            "cumul_number_single_memories_used_next_target_network_update"
                        ] += misc.number_memories_trained_on_between_target_network_updates
                        print("UPDATE")
                        nn_utilities.soft_copy_param(model2, model1, misc.soft_update_tau)
                        # model2.load_state_dict(model.state_dict())
        #buffer.sync_prefetching()  # Finish all prefetching to avoid invalid prefetches during rollouts where the pinned image buffer will be overwritten
        print("")
    
        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY NOW AND THEN
        # ===============================================
        if True:
            accumulated_stats["cumul_training_hours"] += (time.time() - time_last_save) / 3600
            time_last_save = time.time()
    
            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                "gamma": misc.gamma,
                "n_steps": misc.n_steps,
                "epsilon": misc.epsilon,
                "epsilon_boltzmann": misc.epsilon_boltzmann,
                "tau_epsilon_boltzmann": misc.tau_epsilon_boltzmann,
                "tau_greedy_boltzmann": misc.tau_greedy_boltzmann,
                "AL_alpha": misc.AL_alpha,
                "learning_rate": misc.learning_rate,
                "discard_non_greedy_actions_in_nsteps": misc.discard_non_greedy_actions_in_nsteps,
                "reward_per_ms_press_forward": misc.reward_per_ms_press_forward,
            }
            if len(loss_history) > 0:
                step_stats.update(
                    {
                        "loss": np.mean(loss_history),
                        "loss_test": np.mean(loss_test_history),
                        "train_on_batch_duration": np.median(train_on_batch_duration_history),
                    }
                )
    
            for key, value in accumulated_stats.items():
                step_stats[key] = value
    
            loss_history = []
            loss_test_history = []
            train_on_batch_duration_history = []
    
            # ===============================================
            #   COLLECT IQN SPREAD
            # ===============================================
    
            tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")
            #state_img_tensor = torch.as_tensor(
            #    np.expand_dims(pinned_buffer.get_frame_from_buffer(rollout_results["frames"][0]), axis=0)
            #).to(  # TODO : remove as_tensor and expand dims, because this is already pinned memory
            #    "cuda", memory_format=torch.channels_last, non_blocking=True
            #)
            #state_float_tensor = torch.as_tensor(
            #    np.expand_dims(
            #        np.hstack(
            #            (
            #                0,
            #                np.array([True, False, False, False]),  # NEW
            #                rollout_results["car_gear_and_wheels"][0].ravel(),  # NEW
            #                rollout_results["car_orientation"][0].T.dot(rollout_results["car_angular_speed"][0]),  # NEW
            #                rollout_results["car_orientation"][0].T.dot(rollout_results["car_velocity"][0]),
            #                rollout_results["car_orientation"][0].T.dot(np.array([0, 1, 0])),
            #                rollout_results["car_orientation"][0]
            #                .T.dot((zone_centers[0 : misc.n_zone_centers_in_inputs, :] - rollout_results["car_position"][0]).T)
            #                .T.ravel(),
            #            )
            #        ).astype(np.float32),
            #        axis=0,
            #    )
            #).to("cuda", non_blocking=True)
    
            # Désactiver noisy, tirer des tau équitablement répartis
            #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            #    with torch.no_grad():
            #        per_quantile_output = model1(state_img_tensor, state_float_tensor, misc.iqn_k, tau=tau)[0]
    
            #for i, std in enumerate(list(per_quantile_output.cpu().numpy().astype(np.float32).std(axis=0))):
            #    step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std
            #model1.train()
    
            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================
    
            walltime_tb = float(accumulated_stats["cumul_training_hours"] * 3600) + time.time() - time_last_save
            for name, param in model1.named_parameters():
                tensorboard_writer.add_scalar(
                    tag=f"layer_{name}_L2",
                    scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
            assert len(optimizer1.param_groups) == 1
            try:
                for p, (name, _) in zip(optimizer1.param_groups[0]["params"], model1.named_parameters()):
                    state = optimizer1.state[p]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    mod_lr = 1 / (exp_avg_sq.sqrt() + 1e-4)
                    # print("exp_avg                : ", np.sqrt((exp_avg**2).mean().detach().cpu().item()))
                    # print("exp_avg_sq             : ", np.sqrt((exp_avg_sq ** 2).mean().detach().cpu().item()))
                    # print("modified_learning_rate            : ", f"{np.sqrt((mod_lr ** 2).mean().detach().cpu().item()):.2f}")
                    tensorboard_writer.add_scalar(
                        tag=f"lr_ratio_{name}_L2",
                        scalar_value=np.sqrt((mod_lr**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_{name}_L2",
                        scalar_value=np.sqrt((exp_avg**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_sq_{name}_L2",
                        scalar_value=np.sqrt((exp_avg_sq**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
            except:
                pass
    
            for k, v in step_stats.items():
                tensorboard_writer.add_scalar(
                    tag=k,
                    scalar_value=v,
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}: {accumulated_stats['alltime_min_ms'] / 1000:.2f}",
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )
    
            # ===============================================
            #   BUFFER STATS
            # ===============================================
    
            mean_in_buffer = np.array([experience.state_float for experience in buffer._storage]).mean(axis=0)
            std_in_buffer = np.array([experience.state_float for experience in buffer._storage]).std(axis=0)
    
            #print("Raw mean in buffer  :", mean_in_buffer.round(1))
            #print("Raw std in buffer   :", std_in_buffer.round(1))
            #print("")
            #print(
            #    "Corr mean in buffer :",
            #    ((mean_in_buffer - misc.float_inputs_mean) / misc.float_inputs_std).round(1),
            #)
            #print("Corr std in buffer  :", (std_in_buffer / misc.float_inputs_std).round(1))
            #print("")
    
            # ===============================================
            #   SAVE
            # ===============================================
    
            torch.save(model1.state_dict(), save_dir / "weights1.torch")
            torch.save(model2.state_dict(), save_dir / "weights2.torch")
            torch.save(optimizer1.state_dict(), save_dir / "optimizer1.torch")
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
    
            # ===============================================
            #   RELOAD
            # ===============================================
            importlib.reload(misc)
    
            # ===============================================
            #   VERY BASIC TRAINING ANNEALING
            # ===============================================
    
            #if accumulated_stats["cumul_number_frames_played"] > 300_000:
            #    misc.reward_per_ms_press_forward = 0
            #    misc.discard_non_greedy_actions_in_nsteps=True
            if accumulated_stats["cumul_number_frames_played"] < 250_000:
                misc.learning_rate *= 5
            elif accumulated_stats["cumul_number_frames_played"] <500_000:
                misc.learning_rate *= 5 * (1-(accumulated_stats["cumul_number_frames_played"] - 500_000 + 250_000 )/250_000)
    
            # ===============================================
            #   RELOAD
            # ===============================================
    
            for param_group in optimizer1.param_groups:
                param_group["lr"] = misc.learning_rate
            trainer.gamma = misc.gamma
            trainer.AL_alpha = misc.AL_alpha
            trainer.tau_epsilon_boltzmann = misc.tau_epsilon_boltzmann
            trainer.tau_greedy_boltzmann = misc.tau_greedy_boltzmann
