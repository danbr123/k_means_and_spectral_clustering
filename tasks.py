from invoke import task

@task
def delete(c):
    c.run("rm -rf *mykmeanssp*.so")

@task(delete)
def run(c,k,n,Random = True):
    c.run("python3.8.5 setup.py build_ext --inplace") #TODO replace 3.7 with 3.8.5
    try:
        K = int(k)
        N = int(n)
    except ValueError:
        print("Error in parameters provided")
        return 0
    c.run("python3.8.5 main.py "+str(K)+" "+str(N)+" "+str(Random)) #TODO replace 3.7 with 3.8.5



