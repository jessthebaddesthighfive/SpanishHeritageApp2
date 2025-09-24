
import streamlit as st
import sympy as sp

st.set_page_config(page_title="Function Analysis App", layout="wide")

st.title("📈 Function Analysis App")

# User input for function
user_input = st.text_input("Enter a function of x:", "x**2 + 2*x + 1")

# Define the symbol
x = sp.symbols("x")

try:
    func = sp.sympify(user_input)
    
    st.subheader("Function Entered:")
    st.latex(sp.latex(func))

    # Derivative
    derivative = sp.diff(func, x)
    st.subheader("Derivative:")
    st.latex(sp.latex(derivative))

    # Second Derivative
    second_derivative = sp.diff(derivative, x)
    st.subheader("Second Derivative:")
    st.latex(sp.latex(second_derivative))

    # Roots
    roots = sp.solve(func, x)
    st.subheader("Roots:")
    st.write(roots)

    # Critical Points
    critical_points = sp.solve(derivative, x)
    st.subheader("Critical Points:")
    st.write(critical_points)

    # Inflection Points
    inflection_points = sp.solve(second_derivative, x)
    st.subheader("Inflection Points:")
    st.write(inflection_points)

except Exception as e:
    st.error(f"Error: {e}")
