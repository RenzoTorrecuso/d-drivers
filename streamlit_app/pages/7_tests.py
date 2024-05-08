import streamlit_shadcn_ui as ui

with ui.card(key="card1"):

    ui.element(
        "span",
        children=[f"TEST"],
        className="text-green-400 text-lg font-bold m-1",
        key="label2",
    )
