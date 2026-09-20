import os

base_dir = os.path.dirname(os.path.abspath(__file__))
src_app = os.path.join(base_dir, "Src", "streamlit_app.py")

# Make streamlit_app.py see its own correct location
streamlit_globals = {
    "__file__": src_app,
    "__name__": "__main__",
}

with open(src_app, "r", encoding="utf-8") as f:
    exec(compile(f.read(), src_app, "exec"), streamlit_globals)