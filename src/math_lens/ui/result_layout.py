"""Shared display-only sizing for the two result-first pages."""

from contextlib import contextmanager

import streamlit as st


@contextmanager
def result_frame():
    """Retain native image controls, with an aspect-safe viewport height cap."""
    st.html("""<style>
    [data-testid='stMainBlockContainer'] {padding-top:3.5rem;}
    .st-key-lens_main_result {width:88%; max-width:1100px; margin-inline:auto;}
    .st-key-lens_main_result [data-testid='stImage'] img {
        max-height:max(180px, min(780px, calc(100dvh - 280px))); object-fit:contain;
    }
    @media (max-width:800px) {.st-key-lens_main_result {width:100%;}}
    </style>""")
    with st.container(key="lens_main_result"):
        yield
