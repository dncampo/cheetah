# Data analysis
- Description: Cheetah Melanoma Detection is a solution to analyse skin lesions and assign a priority score to schedule an appointment with the specialist based on the three most probable skin lesions detected by the model.
- Data Source: HAM10000 dataset from ISIC archive
- Type of analysis: Deep Learning binary and multiclass models.


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for ham10k-wagon in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/ham10k-wagon`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "ham10k-wagon"
git remote add origin git@github.com:{group}/ham10k-wagon.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
ham10k-wagon-run
```

# Install

Go to `https://github.com/{group}/ham10k-wagon` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/ham10k-wagon.git
cd ham10k-wagon
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
ham10k-wagon-run
```
