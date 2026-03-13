import numpy as np
from typing import Union

# Conversion constant from Kelvin to Celsius
KELVIN_TO_CELSIUS = 273.16

ArrayLike = Union[float, np.ndarray]

def esatw(t_k: ArrayLike) -> ArrayLike:
    """
    Calculates the saturation vapor pressure over water using a polynomial fit.
    This function is vectorized to handle scalar or numpy array inputs.

    Args:
        t_k (Union[float, np.ndarray]): Temperature in Kelvin (K).

    Returns:
        Union[float, np.ndarray]: Saturation vapor pressure in millibars (mb).
    """
    # Coefficients for the polynomial approximation, in descending order of power.
    coefficients = [
        -0.976195544e-15,  # dt^8
        -0.952447341e-13,  # dt^7
        0.640689451e-10,   # dt^6
        0.206739458e-7,    # dt^5
        0.302950461e-5,    # dt^4
        0.264847430e-3,    # dt^3
        0.142986287e-1,    # dt^2
        0.443987641,       # dt^1
        6.11239921,        # dt^0
    ]

    # Temperature difference from freezing point, capped at -80 C.
    t_c = t_k - KELVIN_TO_CELSIUS
    dt = np.maximum(-80.0, t_c)

    # Evaluate the polynomial. np.polyval handles both scalar and array inputs.
    return np.polyval(coefficients, dt)

def qsatw(t_k: ArrayLike, p_mb: ArrayLike) -> ArrayLike:
    """
    Calculates the saturation specific humidity over water.
    This function is vectorized to handle scalar or numpy array inputs.

    Args:
        t_k (Union[float, np.ndarray]): Temperature in Kelvin (K).
        p_mb (Union[float, np.ndarray]): Pressure in millibars (mb).

    Returns:
        Union[float, np.ndarray]: Saturation specific humidity (dimensionless, kg/kg).
    """
    esat = esatw(t_k)
    
    # np.maximum is element-wise, handling both scalar and array inputs.
    return 0.622 * esat / np.maximum(esat, p_mb - esat)

if __name__ == '__main__':
    # --- Scalar Test ---
    temp_scalar = 288.15  # 15 degrees C
    pressure_scalar = 1013.25
    print("--- Scalar Test ---")
    print(f"Input Temperature: {temp_scalar:.2f} K")
    print(f"Input Pressure: {pressure_scalar:.2f} mb")
    print(f"Saturation Vapor Pressure: {esatw(temp_scalar):.4f} mb")
    print(f"Saturation Specific Humidity: {qsatw(temp_scalar, pressure_scalar):.6f} kg/kg")
    print("-" * 25)

    # --- Array Test ---
    temperatures_k = np.array([273.15, 288.15, 293.15, 303.15]) # 0, 15, 20, 30 C
    pressures_mb = np.array([1000.0, 1010.0, 1015.0, 1020.0])

    esat_values = esatw(temperatures_k)
    qsat_values = qsatw(temperatures_k, pressures_mb)

    print("\n--- Array Test ---")
    print(f"Input Temperatures (K): {temperatures_k}")
    print(f"Input Pressures (mb): {pressures_mb}")
    print("\n--- Results ---")
    for i in range(len(temperatures_k)):
        print(f"  For T={temperatures_k[i]:.2f}K, P={pressures_mb[i]:.2f}mb:")
        print(f"    esatw = {esat_values[i]:.4f} mb")
        print(f"    qsatw = {qsat_values[i]:.6f} kg/kg")
    print("-" * 25)
