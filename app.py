import numpy as np
import sympy as sp
import pandas as pd
import streamlit as st
from methods import bisection_method

st.header("Numerical Methods")

bisection_input = st.toggle("Bisection Method", False)
if bisection_input:
    st.write("##### Finding the root of the function using the Bisection Method")

    def bisection_input():
        function = st.sidebar.text_input("Enter the function in terms of x (e.g. x ** 2 - 4)", value="x ** 3 - x ** 2 + 2")
        a = st.sidebar.slider("Enter the value of a", -1000, 1000, -200)
        b = st.sidebar.slider("Enter the value of b", -1000, 1000, 300)
        error = st.sidebar.slider("Enter the error", 0.00, 1.0, 0.001)
        max_iterations = st.sidebar.slider("Enter the maximum number of iterations", 1, 10000, 100)
        
        return function, a, b, error, max_iterations
    function, a, b, error, max_iterations = bisection_input()


    st.write(f"Function: {function}", )
    st.write(f"a: {a}")
    st.write(f"b: {b}")
    st.write(f"Error: {error}")


    root, table_of_iterations = bisection_method(a, b, function, error, max_iterations)

    if root is None:
        st.write("Root not found")
    else :    
        st.write("#### Table of iterations", table_of_iterations)
        st.write("#### Root ",root)
    
