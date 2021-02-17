from invoke import task

@task
def delete(c):
    c.run("rm -rf *mykmeanssp*.so")
    print("Done cleaning")

@task(delete)
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")

    print("Done building")


