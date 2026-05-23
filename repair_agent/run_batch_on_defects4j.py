import multiprocessing
import subprocess

def run_bash_script(param):
    try:
        save_file = open("batch_log_" + str(param[-1]), "a")
        err_file = open("batch_err_" + str(param[-1]), "a")
        print("saving_to_file:", "batch_log_" + str(param[-1]), "batch_err_" + str(param[-1]))

        subprocess.run(['bash', './run_on_defects4j.sh', *param[:-1]], check=True, stdout=save_file, stderr=err_file)
        return multiprocessing.current_process().name
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return None

if __name__ == '__main__':
    # Model selection is driven by SMART_LLM / FAST_LLM / STATIC_LLM env vars
    # (or by --smart-model / --fast-model / --static-model on direct run.sh
    # invocations). To use a single model for all roles instead, append it as
    # a 3rd positional arg, e.g. ("…/batches/0", "hyperparams.json", "gpt-4o-mini", i).
    params_list = [("experimental_setups/batches/{}".format(i), "hyperparams.json", i) for i in range(20)]
    finished_processes = []

    with multiprocessing.Pool() as pool:
        results = pool.map(run_bash_script, params_list)
        for result in results:
            if result:
                finished_processes.append(result)

    print("Finished processes:", finished_processes)