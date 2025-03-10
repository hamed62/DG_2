import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import io
import base64

def main():
    st.title("💩💩K-Map Generator💩💩")
    
    # Step 1: User inputs number of inputs and outputs
    col1, col2 = st.columns(2)
    with col1:
        num_inputs = st.number_input("Number of input variables", min_value=1, max_value=4, value=4)
    with col2:
        num_outputs = st.number_input("Number of output variables", min_value=1, max_value=4, value=4)
    
    # Generate truth table
    input_combinations = generate_input_combinations(num_inputs)
    
    # Create truth table DataFrame
    truth_table = create_truth_table(input_combinations, num_inputs, num_outputs)
    
    # Display interactive truth table
    updated_truth_table = display_interactive_table(truth_table, num_inputs, num_outputs)
    
    # Generate K-maps for each output
    st.header("------------------------------------------------------------------------")
    
    for i in range(num_outputs):
        output_col = f"O{i+1}"
        st.subheader(f"K-Map for {output_col}")
        
        # Generate K-map
        k_map = generate_kmap(updated_truth_table, num_inputs, output_col)
        
        # Display interactive K-map
        display_interactive_kmap(k_map, num_inputs, output_col)
        
        
    # Simplified expressions
    st.header("FINAL Expressions")
    for i in range(num_outputs):
        output_col = f"O{i+1}"
        equation_key = f"equation_{output_col}"
        
        # Get the equation from session state
        equation = st.session_state.get(equation_key, "")
        
        # Display the equation
        if equation:
            st.write(f"{output_col} = {equation}")
        else:
            st.write(f"{output_col} = (No expression entered)")

def generate_input_combinations(num_inputs):
    """Generate all possible input combinations"""
    rows = 2 ** num_inputs
    return [format(i, f'0{num_inputs}b') for i in range(rows)]

def create_truth_table(input_combinations, num_inputs, num_outputs):
    """Create initial truth table DataFrame"""
    rows = len(input_combinations)
    
    # Create DataFrame
    df = pd.DataFrame()
    
    # Add row numbers
    df[''] = list(range(rows))
    
    # Add input columns
    input_cols = ['A', 'B', 'C', 'D'][:num_inputs]
    for i, col in enumerate(input_cols):
        df[col] = [int(combo[i]) for combo in input_combinations]
    
    # Add output columns (initialized to 0)
    output_cols = [f'O{i+1}' for i in range(num_outputs)]
    for col in output_cols:
        df[col] = 0
        
    return df

def display_interactive_table(df, num_inputs, num_outputs):
    """Display an interactive truth table where outputs can be toggled"""
    st.header("Truth Table")
    
    # Initialize session state for the truth table if not already done
    if "truth_table_values" not in st.session_state:
        st.session_state.truth_table_values = {}
        for idx, row in df.iterrows():
            for i in range(num_outputs):
                output_col = df.columns[num_inputs + 1 + i]
                key = f"{idx}_{output_col}"
                st.session_state.truth_table_values[key] = 0
    
    # Create a copy of the dataframe to store the updated values
    updated_df = df.copy()
    
    # Update the dataframe with the current session state values
    for idx, row in df.iterrows():
        for i in range(num_outputs):
            output_col = df.columns[num_inputs + 1 + i]
            key = f"{idx}_{output_col}"
            updated_df.at[idx, output_col] = st.session_state.truth_table_values[key]
    
    # Display the table with clickable cells for outputs
    cols = st.columns([1] + [1] * num_inputs + [1] * num_outputs)
    
    # Table headers
    for i, col in enumerate(cols):
        if i == 0:
            col.markdown("<div style='text-align: center; font-weight: bold;'></div>", unsafe_allow_html=True)
        elif i <= num_inputs:
            col.markdown(f"<div style='text-align: center; font-weight: bold;'>{df.columns[i]}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align: center; font-weight: bold;'>{df.columns[i]}</div>", unsafe_allow_html=True)
    
    # Table rows
    for idx, row in df.iterrows():
        row_cols = st.columns([1] + [1] * num_inputs + [1] * num_outputs)
        
        # Row number and input values (not clickable)
        for i in range(num_inputs + 1):
            if i == 0:
                row_cols[i].markdown(f"<div style='text-align: center;'>{row['']}</div>", unsafe_allow_html=True)
            else:
                row_cols[i].markdown(f"<div style='text-align: center;'>{row[df.columns[i]]}</div>", unsafe_allow_html=True)
        
        # Output values (clickable)
        for i in range(num_outputs):
            output_col = df.columns[num_inputs + 1 + i]
            col_idx = num_inputs + 1 + i
            
            # Use a unique key for each button
            key = f"{idx}_{output_col}"
            
            # Get current value from session state
            current_val = st.session_state.truth_table_values[key]
            
            # Create the toggle button
            if row_cols[col_idx].button(
                f"{current_val}", 
                key=f"btn_{key}",
                help=f"Click to toggle {output_col} value",
                use_container_width=True,
                type="secondary" if current_val == 0 else "primary"
            ):
                # Toggle the value in session state
                st.session_state.truth_table_values[key] = 1 - current_val
                st.rerun()
            
            # Update the dataframe with the current value
            updated_df.at[idx, output_col] = current_val
    
    # Display the full updated dataframe as a reference
    with st.expander("View Full Truth Table"):
        st.dataframe(updated_df)
    
    return updated_df
    
    # Display the full updated dataframe as a reference
    with st.expander("View Full Truth Table"):
        st.dataframe(updated_df)
    
    return updated_df

def generate_kmap(df, num_inputs, output_col):
    """Generate K-map values for the specified output"""
    if num_inputs == 1:
        # 1-input K-map logic remains the same
        kmap = np.zeros((1, 2), dtype=int)
        kmap[0, 0] = df.loc[df['A'] == 0, output_col].values[0]
        kmap[0, 1] = df.loc[df['A'] == 1, output_col].values[0]
    
    elif num_inputs == 2:
        # 2-input K-map logic remains the same
        kmap = np.zeros((1, 4), dtype=int)
        kmap[0, 0] = df.loc[(df['A'] == 0) & (df['B'] == 0), output_col].values[0]
        kmap[0, 1] = df.loc[(df['A'] == 0) & (df['B'] == 1), output_col].values[0]
        kmap[0, 2] = df.loc[(df['A'] == 1) & (df['B'] == 0), output_col].values[0]
        kmap[0, 3] = df.loc[(df['A'] == 1) & (df['B'] == 1), output_col].values[0]
    
    elif num_inputs == 3:
        # 3-input K-map logic remains the same
        kmap = np.zeros((2, 4), dtype=int)
        # Row 0 (A=0)
        kmap[0, 0] = df.loc[(df['A'] == 0) & (df['B'] == 0) & (df['C'] == 0), output_col].values[0]
        kmap[0, 1] = df.loc[(df['A'] == 0) & (df['B'] == 0) & (df['C'] == 1), output_col].values[0]
        kmap[0, 3] = df.loc[(df['A'] == 0) & (df['B'] == 1) & (df['C'] == 0), output_col].values[0]
        kmap[0, 2] = df.loc[(df['A'] == 0) & (df['B'] == 1) & (df['C'] == 1), output_col].values[0]
        # Row 1 (A=1)
        kmap[1, 0] = df.loc[(df['A'] == 1) & (df['B'] == 0) & (df['C'] == 0), output_col].values[0]
        kmap[1, 1] = df.loc[(df['A'] == 1) & (df['B'] == 0) & (df['C'] == 1), output_col].values[0]
        kmap[1, 3] = df.loc[(df['A'] == 1) & (df['B'] == 1) & (df['C'] == 0), output_col].values[0]
        kmap[1, 2] = df.loc[(df['A'] == 1) & (df['B'] == 1) & (df['C'] == 1), output_col].values[0]
    
    elif num_inputs == 4:
        kmap = np.zeros((4, 4), dtype=int)
        
        # Row and column mappings for K-map (Gray code order)
        # AB values for rows: 00, 01, 11, 10
        # CD values for columns: 00, 01, 11, 10
        
        # This matches the minterm ordering: 0,1,3,2,4,5,7,6,12,13,15,14,8,9,11,10
        
        # Map from row/col position to minterm binary
        positions = [
            # Row 0: AB=00, (minterms 0,1,3,2)
            [(0,0,0,0), (0,0,0,1), (0,0,1,1), (0,0,1,0)],
            # Row 1: AB=01, (minterms 4,5,7,6)
            [(0,1,0,0), (0,1,0,1), (0,1,1,1), (0,1,1,0)],
            # Row 2: AB=11, (minterms 12,13,15,14)
            [(1,1,0,0), (1,1,0,1), (1,1,1,1), (1,1,1,0)],
            # Row 3: AB=10, (minterms 8,9,11,10)
            [(1,0,0,0), (1,0,0,1), (1,0,1,1), (1,0,1,0)]
        ]
        
        # Fill the K-map with values from the truth table
        for i in range(4):
            for j in range(4):
                a, b, c, d = positions[i][j]
                idx = df.loc[(df['A'] == a) & (df['B'] == b) & 
                             (df['C'] == c) & (df['D'] == d)].index[0]
                kmap[i, j] = df.loc[idx, output_col]
    
    return kmap


def display_interactive_kmap(kmap, num_inputs, output_col):
    """Display an interactive K-map that allows grouping minterms"""
    # Determine the K-map dimensions
    if num_inputs == 1:
        rows, cols = 1, 2
        row_labels = ['']
        col_labels = ['0', '1']
    elif num_inputs == 2:
        rows, cols = 1, 4
        row_labels = ['']
        col_labels = ['00', '01', '11', '10']
    elif num_inputs == 3:
        rows, cols = 2, 4
        row_labels = ['0', '1'] 
        col_labels = ['00', '01', '11', '10']
    else:  # num_inputs == 4
        rows, cols = 4, 4
        row_labels = ['00', '01', '11', '10']
        col_labels = ['00', '01', '11', '10']
    
    # Initialize session state for kmap groups if not already done
    group_key = f"kmap_groups_{output_col}"
    if group_key not in st.session_state:
        st.session_state[group_key] = []  # List to store all groups
    
    # Initialize the selected color in session state
    color_key = f"kmap_color_{output_col}"
    if color_key not in st.session_state:
        st.session_state[color_key] = "red"  # Default color
    
    # Color selection UI
    st.write("Select a color for grouping:")
    color_cols = st.columns(4)
    colors = ["red", "blue", "green", "purple"]
    
    for i, color in enumerate(colors):
        if color_cols[i].button(f"{color.capitalize()}", key=f"color_{output_col}_{color}"):
            st.session_state[color_key] = color
            st.rerun()
    
    st.write(f"Current color: {st.session_state[color_key]}")
    
    # Create a visual representation of the K-map that's clickable
    st.write("Click on cells to create/edit groups:")
    
    # Initialize session state for cell selections if not already done
    cells_key = f"selected_cells_{output_col}"
    if cells_key not in st.session_state:
        st.session_state[cells_key] = []
    
    # Draw K-map as a grid of buttons
    for i in range(rows):
        cols_ui = st.columns(cols)
        for j in range(cols):
            cell_value = kmap[i, j]
            cell_id = f"{i}_{j}"
            
            # Check if this cell is in any groups
            cell_colors = []
            for group in st.session_state[group_key]:
                if (i, j) in group["cells"]:
                    cell_colors.append(group["color"])
            
            # Determine button color and style
            button_style = ""
            if cell_value == 1:
                bg_color = "yellow"
                if cell_colors:
                    # If cell is in groups, show a border indicator
                    borders = []
                    for color in cell_colors:
                        borders.append(f"3px solid {color}")
                    button_style = f"border: {'; '.join(borders)};"
            else:
                bg_color = "white"
                if cell_colors:
                    # If cell is in groups, show a border indicator
                    borders = []
                    for color in cell_colors:
                        borders.append(f"3px solid {color}")
                    button_style = f"border: {'; '.join(borders)};"
            
            # Create a clickable button for each cell
            button_label = f"{cell_value}"
            if cols_ui[j].button(
                button_label,
                key=f"cell_{output_col}_{i}_{j}",
                help=f"Click to select cell ({i},{j})",
                use_container_width=True,
                type="primary" if cell_value == 1 else "secondary",
            ):
                # Toggle this cell in the selected cells list
                if (i, j) not in st.session_state[cells_key]:
                    st.session_state[cells_key].append((i, j))
                else:
                    st.session_state[cells_key].remove((i, j))
                st.rerun()
    
    # Display currently selected cells
    if st.session_state[cells_key]:
        st.write(f"Selected cells: {st.session_state[cells_key]}")
        
        # Add a button to create a group from selected cells
        if st.button("Create Group", key=f"create_group_{output_col}"):
            if st.session_state[cells_key]:
                # Add new group
                st.session_state[group_key].append({
                    "cells": st.session_state[cells_key].copy(),
                    "color": st.session_state[color_key]
                })
                # Clear selection
                st.session_state[cells_key] = []
                st.rerun()
    
    # Display current groups
    if st.session_state[group_key]:
        st.write("Current Groups:")
        for i, group in enumerate(st.session_state[group_key]):
            col1, col2 = st.columns([4, 1])
            col1.write(f"Group {i+1}: {len(group['cells'])} cells in {group['color']}")
            if col2.button("Delete", key=f"delete_group_{output_col}_{i}"):
                st.session_state[group_key].pop(i)
                st.rerun()
    
    # Button to clear all groups
    if st.button("Clear All Groups", key=f"clear_all_{output_col}"):
        st.session_state[group_key] = []
        st.rerun()
    
    # Also create a matplotlib visualization for reference and potential export
    fig, ax = plt.subplots(figsize=(6, 3 if rows <= 2 else 6))
    ax.set_title(f"K-map for {output_col}")
    
    # Plot the K-map
    for i in range(rows):
        for j in range(cols):
            # Basic cell
            ax.text(j + 0.5, i + 0.5, str(kmap[i, j]), 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14)
            rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Highlight cells with value 1
            if kmap[i, j] == 1:
                rect = Rectangle((j, i), 1, 1, fill=True, facecolor='yellow', 
                                 alpha=0.3, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
    
    # Draw groups from session state
    for group in st.session_state[group_key]:
        color = group["color"]
        cells = group["cells"]
        
        # For each group, draw a colored rectangle that encompasses all cells
        if cells:
            min_i = min(i for i, j in cells)
            max_i = max(i for i, j in cells)
            min_j = min(j for i, j in cells)
            max_j = max(j for i, j in cells)
            
            width = max_j - min_j + 1
            height = max_i - min_i + 1
            
            rect = Rectangle((min_j, min_i), width, height, fill=False, 
                             edgecolor=color, linewidth=3, linestyle='-')
            ax.add_patch(rect)
    
    # Add row and column labels
    for i, label in enumerate(row_labels):
        ax.text(-0.3, i + 0.5, label, horizontalalignment='center', 
                 verticalalignment='center', fontsize=12)
    
    for j, label in enumerate(col_labels):
        ax.text(j + 0.5, -0.3, label, horizontalalignment='center', 
                 verticalalignment='center', fontsize=12)
    
    # Add input labels
    if num_inputs <= 2:
        ax.text(-0.7, -0.3, "A", fontsize=12)
        if num_inputs == 2:
            ax.text(2, -0.7, "B", fontsize=12)
    else:
        ax.text(-0.7, rows/2, "A", fontsize=12)
        ax.text(cols/2, -0.7, "CD" if num_inputs == 4 else "C", fontsize=12)
        if num_inputs == 4:
            ax.text(-1.1, rows/2, "AB", fontsize=12)
    
    # Set limits and remove ticks
    ax.set_xlim(-1, cols)
    ax.set_ylim(rows, -1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set aspect ratio and adjust layout
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Convert the figure to a base64 string and display it
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    st.write("K-Map Visualization (non-interactive):")
    st.image(f"data:image/png;base64,{img_str}")
    
    # Ensure session state is initialized before creating the text input field
    equation_key = f"equation_{output_col}"

    if equation_key not in st.session_state:
        st.session_state[equation_key] = ""

    st.write("Enter your simplified Boolean expression:")

    # Text input field
    user_equation = st.text_input(
        label=f"Simplified expression for {output_col}",
        value=st.session_state[equation_key],
        key=equation_key
    )

    # Instead of modifying st.session_state[equation_key] after widget creation,
    # We store it conditionally only when the user updates the input.
    if user_equation != st.session_state[equation_key]:  
        st.session_state[equation_key] = user_equation  # Update session state

    st.write("---")

if __name__ == "__main__":
    st.set_page_config(page_title="K-Map Generator", layout="wide")
    
    # Initialize session state for groups if not already done
    if "groups" not in st.session_state:
        st.session_state.groups = {}
    
    main()