# Playground Instruction

Demo scripy is saved in `.py` format but it can be quickly converted into
`.ipynotebook` by running the following script under the current directory
leveraging `jupytext`.

```
bash convert_to_notebook.sh
```

Or you can directly run the following make the convert to notebook.

```
jupytext --to notebook demo.py
```

And if you wish to update `py` from notebook, you can simply run the code below.

```
jupytext --to py demo.ipynb
```
