from invoke import task

@task
def delete(c):
    c.run("rm -rf *mykmeanssp*.so")

@task(delete)
def run(c,k,n,Random = True):
    print(k,n,Random)
    c.run("python3.7 setup.py build_ext --inplace")
    try:
        K = int(k)
        N = int(n)
    except ValueError:
        print("Error in parameters provided")
        return 0
    try:
        r = int(Random)
    except ValueError:
        if(Random!="False"):
            print("Error in parameters provided")
            return 0
        r = False

    c.run("python3.7 main.py "+str(K)+" "+str(N)+" "+str(r))

    #c.run("python3.8.5 setup.py build_ext --inplace")


