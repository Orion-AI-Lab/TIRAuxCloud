from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import sys
import time
import traceback

def format_time(elapsed):
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"

def run_in_parallel(func, datachunks, args=(), max_workers=None, show_progress=True, skipnone=True, print_elapsed=True):
    """
    Run `func(file, *args)` in parallel for each file in `files`.

    Parameters:
        func         : Function to call, must take at least one argument (file), then any number of additional arguments.
        datachunks        : List of work items (e.g., file paths).
        args         : List or tuple of additional arguments passed to the function.
        max_workers  : Number of parallel processes (defaults to os.cpu_count()).
        show_progress : show progress bar
        skipnone: do not append to results if output is none or empty

    Returns:
        List of non-empty results returned by `func`.
    """

    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, chunk, *args) for chunk in datachunks]
        total = len(futures)
        if show_progress:
            pbar = tqdm.tqdm(total=total, desc="Processing", file=sys.stdout)
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if skipnone and result:
                    results.append(result)
                elif not skipnone:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] Exception in worker:")
                traceback.print_exc()
            finally:
                if show_progress:
                    pbar.update(1)
        '''
        for future in as_completed(futures):
            result = future.result()
            if skipnone and result:  # Skip None or empty
                results.append(result)
            elif not skipnone:
                results.append(result)
            if show_progress:
                pbar.update(1)
                sys.stdout.flush()
        '''
        if show_progress:
            pbar.close()

    end_time = time.time()
    elapsed = end_time - start_time
    if print_elapsed:
        print(f"Execution time: {format_time(elapsed)}")
  
    return results

### example usage ###
# def my_worker(file, arg1, arg2):
    # Dummy example
#    return {"file": file, "sum": len(file) + arg1 + arg2}

# files = ["file1", "file2", "file3"]
# extra_args = [10, 5]

# results = run_in_parallel(my_worker, files, extra_args)
# print(results)

