from invoke import task

''' 
    task to delete *mykmeanssp*.so files generated by previous build task
'''
@task
def delete(c):
    c.run("rm -rf *mykmeanssp*.so")

''' 
    task to:
     - build mykmeansssp.cpyton file - python C API - C compiled code files 
     - call and run main.py with arguments pass from user
'''
@task(delete)
def run(c, k=0, n=0, Random=True):
    c.run("python3.8.5 setup.py build_ext --inplace")
    try:
        K = int(k)
        N = int(n)
    except ValueError:
        print("Illegal parameters provided")
        return 0
    c.run("python3.8.5 main.py " + str(K) + " " + str(N) + " " + str(Random))
