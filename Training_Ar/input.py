import os
import math
import numpy as np
from math import sqrt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d  # Import Gaussian filter

# Definition of initial constants
INITIAL_VALUES = {
    "μref": 2.117e-5,
    "Tref": 273.15,
    "dref": 4.11,  # e-10 m
    "Mref": 39.948,
    "ω": 0.81
}

required_vars = ["x", "Ttrans_N", "p_N",
                 "sigma1_N", "q1_N", "ρ_deriv_N",
                 "Ttrans_deriv_N", "p_deriv_N", "sigma1_deriv_N",
                 "q1_deriv_N", "mv1v1v1_mod_N", "mvvv1v1_mod_N",
                 "S4_mod_N", "S5_mod_N"]

# Function for reading data
def read_data(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()[3:]  # Skip the first three lines
        data = [list(map(float, line.split())) for line in lines]
        return np.array(data)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

# Function for cubic interpolation and insertion of additional points
def interpolate_and_insert_points(data, num_new_points=3):
    x = data[:, 0]
    y = data[:, 1:]
    interpolated_data = []

    for i in range(1, len(x) - 2):
        x_window = x[i-1:i+3]
        y_window = y[i-1:i+3]

        cs_list = [CubicSpline(x_window, y_window[:, j]) for j in range(y.shape[1])]
        x_new = np.linspace(x[i], x[i+1], num_new_points + 2)[1:-1]
        y_new = np.column_stack([cs(x_new) for cs in cs_list])

        if i == 1:
            interpolated_data.append(data[0])
        interpolated_data.append(data[i])
        interpolated_data.extend(np.column_stack([x_new, y_new]))

    interpolated_data.append(data[-2])
    interpolated_data.append(data[-1])
    return np.array(interpolated_data)

# Calculate derivatives and normalized quantities, with filtering applied
def calculate_derivatives(data):
    try:
        μref, Tref, dref, Mref, ω = INITIAL_VALUES["μref"], INITIAL_VALUES["Tref"], INITIAL_VALUES["dref"], INITIAL_VALUES["Mref"], INITIAL_VALUES["ω"]

        x, ρ, v1, Ttrans, p, p1, q1, mv1v1v1, mvvv1v1 = (
            data[:, i] for i in range(1, 10)
        )

        vref = sqrt(Tref * 8314.5 / Mref)

        ρ_deriv = np.gradient(ρ, x)
        Ttrans_deriv = np.gradient(Ttrans, x)
        p_deriv = np.gradient(p, x)
        p1_deriv = np.gradient(p1, x)
        q1_deriv = np.gradient(q1, x)

        sigma1 = p1 - p
        sigma1_deriv = p1_deriv - p_deriv

        mfp = (Mref * 1.66053906660e-7)/(sqrt(2)*math.pi*dref**2*(Tref/Ttrans)**(ω-0.5)*ρ)
        μ = μref * (Ttrans/Tref)**ω

        theta = p / ρ

        μ = μref * (Ttrans / Tref) ** ω

        v1_N = v1 / vref
        Ttrans_N = Ttrans / Tref
        p_N = p / (ρ * vref**2)
        sigma1_N = sigma1 / (ρ * vref**2)
        q1_N = q1 / (ρ * vref**3)

        ρ_deriv_N = ρ_deriv / (ρ / mfp)
        Ttrans_deriv_N = Ttrans_deriv / (Tref / mfp)
        p_deriv_N = p_deriv / (ρ * vref**2 / mfp)
        sigma1_deriv_N = sigma1_deriv / (ρ * vref**2 / mfp)
        q1_deriv_N = q1_deriv / (ρ * vref**3 / mfp)

        f4 = ρ * v1**3 + 3.0 * p1 * v1 + mv1v1v1
        f5 = (ρ * v1**2 + 3.0 * p + 4.0 * p1) * v1**2 + v1**2 * p1 + 4.0 * q1 * v1 + 2.0 * mv1v1v1 * v1 + mvvv1v1

        mv1v1v1_mod_N = (mv1v1v1 - 1.2 * q1) / (μ * vref**2 / mfp * 1.0)
        mvvv1v1_mod_N = (mvvv1v1 - 7.0 * theta * p1 + 2.0 * theta * p) / (μ * vref**3 / mfp * 10.0)

        mv1v1v1_mod_N = gaussian_filter1d(mv1v1v1_mod_N, sigma=3)
        mvvv1v1_mod_N = gaussian_filter1d(mvvv1v1_mod_N, sigma=3)

        # Calculate the raw S4 and S5 terms
        S4_mod_N = np.gradient(f4, x) / (p/μ * ρ * vref**2)
        S5_mod_N = (np.gradient(f5, x) - 2 * v1 * np.gradient(f4, x)) / (p/μ * ρ * vref**3 * 10.0)

        # Apply Gaussian filtering
        S4_mod_N = gaussian_filter1d(S4_mod_N, sigma=1)
        S5_mod_N = gaussian_filter1d(S5_mod_N, sigma=1)

        return np.column_stack([
            x, Ttrans_N, p_N, sigma1_N, q1_N,
            ρ_deriv_N, Ttrans_deriv_N,
            p_deriv_N, sigma1_deriv_N, q1_deriv_N,
            mv1v1v1_mod_N, mvvv1v1_mod_N, S4_mod_N, S5_mod_N
        ])
    except Exception as e:
        print(f"Error in derivative calculation: {e}")
        return None

# Function for saving the results
def save_results(output_filename, combined_data):
    try:
        with open(output_filename, 'w') as file:
            file.write('TITLE = "DSMC"\n')
            file.write('VARIABLES = ' + ' '.join([f'"{var}"' for var in required_vars]) + '\n')
            for row in combined_data:
                file.write(' '.join([f"{val:.10e}" for val in row]) + '\n')
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")

# Main function
def main():
    for root, _, files in os.walk('.'):
        for filename in files:
            if filename.endswith("_input.dat"):
                continue

            if filename.startswith(("fwd_", "inv_")) and filename.endswith(".dat"):
                try:
                    file_number = int(filename.split("_")[1].split(".")[0])
                    if 10000 < file_number:
                        filepath = os.path.join(root, filename)
                        output_filename = filepath.replace(".dat", "_input.dat")

                        data = read_data(filepath)
                        if data is not None:
                            combined_data = calculate_derivatives(data)
                            if combined_data is not None:
                                interpolated_data = interpolate_and_insert_points(combined_data)
                                save_results(output_filename, interpolated_data)
                                print(f"Processed {filename} -> {output_filename}")
                except ValueError:
                    continue

if __name__ == "__main__":
    main()

