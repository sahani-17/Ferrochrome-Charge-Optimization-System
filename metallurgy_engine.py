"""
Metallurgy Engine - Ferrochrome Optimization System

This module implements all mathematical formulas for:
- Alloy composition calculations (Cr%, Si%, C%, P%, S%)
- Slag composition calculations (Cr2O3%, FeO%, SiO2%, etc.)
- Process ratios (Basicity, Slag Ratio, etc.)
- Constraint validation (58/4 grade metal + slag specs)
- Cost calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

# ============================================================================
# HARD CONSTRAINTS - PRODUCTION GRADE (58/4 GRADE)
# ============================================================================

ALLOY_CONSTRAINTS = {
    'Cr_min': 57.5,      # 58 ± 0.5%
    'Cr_max': 58.5,
    'Si_max': 4.0,       # Max 4%
    'C_max': 8.5,        # Max 8.5%
    'P_max': 0.03,       # Max 0.03%
    'S_max': 0.05        # Max 0.05%
}

SLAG_CONSTRAINTS = {
    'Cr2O3_max': 8.0,    # Max 8%
    'FeO_max': 2.0,      # Max 2%
    'SiO2_max': 28.0,    # Max 28%
    'Al2O3_max': 25.0,   # Max 25%
    'CaO_max': 8.0,      # Max 8%
    'MgO_max': 27.0      # Max 27%
}

RATIO_CONSTRAINTS = {
    'Basicity_target': 1.25,
    'Basicity_tolerance': 0.05,      # ±5%
    'Slag_Ratio_target': 1.15,
    'Slag_Ratio_tolerance': 0.05     # ±5%
}

# MATERIAL_CONSTRAINTS (per MT of alloy) - OPERATIONAL REQUIREMENTS
# Note: Dataset contains values in MT, multiply by 1000 to get kg
# User specified minimum constraints for optimization
MATERIAL_CONSTRAINTS = {
    'Ore_min': 2038.0,       # kg/MT (minimum for Ore/BRQT)
    'Ore_max': 2500.0,       # kg/MT (upper operational limit)
    'Flux_min': 370.0,       # kg/MT (minimum)
    'Flux_max': 500.0,       # kg/MT (upper operational limit)
    'Reductant_min': 510.0,  # kg/MT (minimum)
    'Reductant_max': 700.0,  # kg/MT (upper operational limit)
    'Auxiliary_min': 11.13,  # kg/MT (Sp Paste minimum)
    'Auxiliary_max': 30.0    # kg/MT (upper operational limit)
}

# ============================================================================
# ALLOY COMPOSITION CALCULATIONS
# ============================================================================

def calculate_alloy_composition(materials_dict: Dict[str, float], 
                                material_master: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate alloy composition based on material inputs.
    
    Uses mathematical formulas:
    - Cr% = (Cr_recovered / (Cr_recovered + Fe_recovered)) * 100
    - Si% = SiO2 * 0.467 * reduction_factor
    - C% = from FC_%
    - P% = P_input * 0.9
    - S% = S_input * 0.85
    
    Args:
        materials_dict: {material_name: quantity_kg}
        material_master: Material composition database
    
    Returns:
        Dictionary with alloy element percentages
    """
    # Initialize totals
    total_Cr = 0.0
    total_Fe = 0.0
    total_SiO2 = 0.0
    total_P = 0.0
    total_S = 0.0
    total_FC = 0.0
    total_mass = sum(materials_dict.values())
    
    # Sum contributions from each material
    for material_name, qty in materials_dict.items():
        material = material_master[material_master['Material_Name'] == material_name].iloc[0]
        
        # Elements (per 100kg basis in material master)
        total_Cr += qty * (material['Cr'] / 100.0)
        total_Fe += qty * (material['Fe'] / 100.0)
        total_SiO2 += qty * (material['SiO2'] / 100.0)
        total_P += qty * (material['P'] / 100.0)
        total_S += qty * (material['S'] / 100.0)
        total_FC += qty * (material['FC_%'] / 100.0)
    
    # Calculate alloy composition
    # Assume 40% of charge becomes alloy (typical for ferrochrome)
    alloy_mass = total_mass * 0.40
    
    # Cr% in alloy (Formula: Cr_recovered / (Cr_recovered + Fe_recovered) * 100)
    Cr_recovered = total_Cr * 0.95  # 95% Cr recovery to alloy
    Fe_recovered = total_Fe * 0.85  # 85% Fe recovery to alloy
    
    if (Cr_recovered + Fe_recovered) > 0:
        Cr_percent = (Cr_recovered / (Cr_recovered + Fe_recovered)) * 100
    else:
        Cr_percent = 0.0
    
    # Si% in alloy (Formula: SiO2 * 0.467 * reduction_factor)
    # Reduction factor ~0.15 (15% of SiO2 reduced to Si)
    Si_percent = (total_SiO2 / alloy_mass) * 0.467 * 0.15 * 100
    
    # C% in alloy (from Fixed Carbon)
    C_percent = (total_FC / alloy_mass) * 100
    
    # P% in alloy (Formula: P_input * 0.9)
    P_percent = (total_P / alloy_mass) * 0.9 * 100
    
    # S% in alloy (Formula: S_input * 0.85)
    S_percent = (total_S / alloy_mass) * 0.85 * 100
    
    # Cr/Fe ratio
    if Fe_recovered > 0:
        Cr_Fe_ratio = Cr_recovered / Fe_recovered
    else:
        Cr_Fe_ratio = 0.0
    
    return {
        'Cr': Cr_percent,
        'Si': Si_percent,
        'C': C_percent,
        'P': P_percent,
        'S': S_percent,
        'Cr_Fe_ratio': Cr_Fe_ratio,
        'Cr_recovered_kg': Cr_recovered,
        'Fe_recovered_kg': Fe_recovered,
        'alloy_mass_kg': alloy_mass
    }

# ============================================================================
# SLAG COMPOSITION CALCULATIONS
# ============================================================================

def calculate_slag_composition(materials_dict: Dict[str, float],
                               material_master: pd.DataFrame,
                               cr_recovery_pct: float) -> Dict[str, float]:
    """
    Calculate slag composition based on material inputs.
    
    Uses mathematical formulas:
    - Slag_Cr2O3 = Cr2O3 * (1 - Cr_recovery)
    - Slag_SiO2 = SiO2 * 0.95
    - Other oxides report to slag
    
    Args:
        materials_dict: {material_name: quantity_kg}
        material_master: Material composition database
        cr_recovery_pct: Chromium recovery percentage
    
    Returns:
        Dictionary with slag oxide percentages
    """
    # Initialize totals
    total_Cr2O3 = 0.0
    total_FeO = 0.0
    total_SiO2 = 0.0
    total_Al2O3 = 0.0
    total_CaO = 0.0
    total_MgO = 0.0
    
    # Sum contributions from each material
    for material_name, qty in materials_dict.items():
        material = material_master[material_master['Material_Name'] == material_name].iloc[0]
        
        # Oxides (per 100kg basis)
        total_Cr2O3 += qty * (material['Cr2O3'] / 100.0)
        total_SiO2 += qty * (material['SiO2'] / 100.0)
        total_Al2O3 += qty * (material['Al2O3'] / 100.0)
        total_CaO += qty * (material['CaO'] / 100.0)
        total_MgO += qty * (material['MgO'] / 100.0)
        # FeO from Fe (Fe -> FeO conversion factor ~1.286)
        total_FeO += qty * (material['Fe'] / 100.0) * 1.286 * 0.15  # 15% Fe to slag
    
    # Calculate slag composition
    # Slag_Cr2O3 = Cr2O3 * (1 - Cr_recovery)
    Slag_Cr2O3 = total_Cr2O3 * (1 - cr_recovery_pct / 100.0)
    
    # Slag_SiO2 = SiO2 * 0.95 (95% reports to slag)
    Slag_SiO2 = total_SiO2 * 0.95
    
    # Other oxides mostly report to slag
    Slag_Al2O3 = total_Al2O3 * 0.98
    Slag_CaO = total_CaO * 0.98
    Slag_MgO = total_MgO * 0.98
    Slag_FeO = total_FeO
    
    # Total slag mass
    total_slag_mass = Slag_Cr2O3 + Slag_SiO2 + Slag_Al2O3 + Slag_CaO + Slag_MgO + Slag_FeO
    
    # Calculate percentages
    if total_slag_mass > 0:
        slag_composition = {
            'Cr2O3': (Slag_Cr2O3 / total_slag_mass) * 100,
            'FeO': (Slag_FeO / total_slag_mass) * 100,
            'SiO2': (Slag_SiO2 / total_slag_mass) * 100,
            'Al2O3': (Slag_Al2O3 / total_slag_mass) * 100,
            'CaO': (Slag_CaO / total_slag_mass) * 100,
            'MgO': (Slag_MgO / total_slag_mass) * 100,
            'total_mass_kg': total_slag_mass
        }
    else:
        slag_composition = {
            'Cr2O3': 0, 'FeO': 0, 'SiO2': 0,
            'Al2O3': 0, 'CaO': 0, 'MgO': 0,
            'total_mass_kg': 0
        }
    
    return slag_composition

# ============================================================================
# PROCESS RATIOS CALCULATIONS
# ============================================================================

def calculate_process_ratios(slag_composition: Dict[str, float],
                             alloy_composition: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate process control ratios.
    
    Formulas:
    - Basicity = (MgO + CaO) / SiO2
    - Slag_Ratio = Slag_mass / Alloy_mass
    - MgO/Al2O3
    - SiO2/(Cr+Fe)
    
    Args:
        slag_composition: Slag oxide percentages
        alloy_composition: Alloy element data
    
    Returns:
        Dictionary with process ratios
    """
    # Basicity = (MgO + CaO) / SiO2
    if slag_composition['SiO2'] > 0:
        basicity = (slag_composition['MgO'] + slag_composition['CaO']) / slag_composition['SiO2']
    else:
        basicity = 0.0
    
    # Slag Ratio = Slag_mass / Alloy_mass
    if alloy_composition['alloy_mass_kg'] > 0:
        slag_ratio = slag_composition['total_mass_kg'] / alloy_composition['alloy_mass_kg']
    else:
        slag_ratio = 0.0
    
    # MgO/Al2O3
    if slag_composition['Al2O3'] > 0:
        MgO_Al2O3 = slag_composition['MgO'] / slag_composition['Al2O3']
    else:
        MgO_Al2O3 = 0.0
    
    # SiO2/(Cr+Fe)
    Cr_Fe_total = alloy_composition['Cr_recovered_kg'] + alloy_composition['Fe_recovered_kg']
    if Cr_Fe_total > 0:
        SiO2_CrFe = slag_composition['total_mass_kg'] * (slag_composition['SiO2']/100) / Cr_Fe_total
    else:
        SiO2_CrFe = 0.0
    
    return {
        'Basicity': basicity,
        'Slag_Ratio': slag_ratio,
        'MgO_Al2O3': MgO_Al2O3,
        'SiO2_CrFe': SiO2_CrFe
    }

# ============================================================================
# CONSTRAINT VALIDATION
# ============================================================================

def validate_constraints(alloy: Dict[str, float],
                        slag: Dict[str, float],
                        ratios: Dict[str, float],
                        material_quantities: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate all hard constraints.
    
    Returns:
        (is_valid, list_of_violations)
    """
    violations = []
    
    # === ALLOY CONSTRAINTS ===
    if alloy['Cr'] < ALLOY_CONSTRAINTS['Cr_min']:
        violations.append(f"Cr% too low: {alloy['Cr']:.2f}% < {ALLOY_CONSTRAINTS['Cr_min']}%")
    if alloy['Cr'] > ALLOY_CONSTRAINTS['Cr_max']:
        violations.append(f"Cr% too high: {alloy['Cr']:.2f}% > {ALLOY_CONSTRAINTS['Cr_max']}%")
    
    if alloy['Si'] > ALLOY_CONSTRAINTS['Si_max']:
        violations.append(f"Si% too high: {alloy['Si']:.2f}% > {ALLOY_CONSTRAINTS['Si_max']}%")
    
    if alloy['C'] > ALLOY_CONSTRAINTS['C_max']:
        violations.append(f"C% too high: {alloy['C']:.2f}% > {ALLOY_CONSTRAINTS['C_max']}%")
    
    if alloy['P'] > ALLOY_CONSTRAINTS['P_max']:
        violations.append(f"P% too high: {alloy['P']:.4f}% > {ALLOY_CONSTRAINTS['P_max']}%")
    
    if alloy['S'] > ALLOY_CONSTRAINTS['S_max']:
        violations.append(f"S% too high: {alloy['S']:.4f}% > {ALLOY_CONSTRAINTS['S_max']}%")
    
    # === SLAG CONSTRAINTS ===
    if slag['Cr2O3'] > SLAG_CONSTRAINTS['Cr2O3_max']:
        violations.append(f"Slag Cr2O3% too high: {slag['Cr2O3']:.2f}% > {SLAG_CONSTRAINTS['Cr2O3_max']}%")
    
    if slag['FeO'] > SLAG_CONSTRAINTS['FeO_max']:
        violations.append(f"Slag FeO% too high: {slag['FeO']:.2f}% > {SLAG_CONSTRAINTS['FeO_max']}%")
    
    if slag['SiO2'] > SLAG_CONSTRAINTS['SiO2_max']:
        violations.append(f"Slag SiO2% too high: {slag['SiO2']:.2f}% > {SLAG_CONSTRAINTS['SiO2_max']}%")
    
    if slag['Al2O3'] > SLAG_CONSTRAINTS['Al2O3_max']:
        violations.append(f"Slag Al2O3% too high: {slag['Al2O3']:.2f}% > {SLAG_CONSTRAINTS['Al2O3_max']}%")
    
    if slag['CaO'] > SLAG_CONSTRAINTS['CaO_max']:
        violations.append(f"Slag CaO% too high: {slag['CaO']:.2f}% > {SLAG_CONSTRAINTS['CaO_max']}%")
    
    if slag['MgO'] > SLAG_CONSTRAINTS['MgO_max']:
        violations.append(f"Slag MgO% too high: {slag['MgO']:.2f}% > {SLAG_CONSTRAINTS['MgO_max']}%")
    
    # === RATIO CONSTRAINTS ===
    basicity_min = RATIO_CONSTRAINTS['Basicity_target'] * (1 - RATIO_CONSTRAINTS['Basicity_tolerance'])
    basicity_max = RATIO_CONSTRAINTS['Basicity_target'] * (1 + RATIO_CONSTRAINTS['Basicity_tolerance'])
    if not (basicity_min <= ratios['Basicity'] <= basicity_max):
        violations.append(f"Basicity out of range: {ratios['Basicity']:.2f} not in [{basicity_min:.2f}, {basicity_max:.2f}]")
    
    slag_ratio_min = RATIO_CONSTRAINTS['Slag_Ratio_target'] * (1 - RATIO_CONSTRAINTS['Slag_Ratio_tolerance'])
    slag_ratio_max = RATIO_CONSTRAINTS['Slag_Ratio_target'] * (1 + RATIO_CONSTRAINTS['Slag_Ratio_tolerance'])
    if not (slag_ratio_min <= ratios['Slag_Ratio'] <= slag_ratio_max):
        violations.append(f"Slag Ratio out of range: {ratios['Slag_Ratio']:.2f} not in [{slag_ratio_min:.2f}, {slag_ratio_max:.2f}]")
    
    # === MATERIAL QUANTITY CONSTRAINTS (RESTORED) ===
    ore_qty = material_quantities.get('Ore_qty', 0)
    if not (MATERIAL_CONSTRAINTS['Ore_min'] <= ore_qty <= MATERIAL_CONSTRAINTS['Ore_max']):
        violations.append(f"Ore quantity out of range: {ore_qty:.0f} kg not in [{MATERIAL_CONSTRAINTS['Ore_min']}, {MATERIAL_CONSTRAINTS['Ore_max']}]")
    
    flux_qty = material_quantities.get('Flux_qty', 0)
    if not (MATERIAL_CONSTRAINTS['Flux_min'] <= flux_qty <= MATERIAL_CONSTRAINTS['Flux_max']):
        violations.append(f"Flux quantity out of range: {flux_qty:.0f} kg not in [{MATERIAL_CONSTRAINTS['Flux_min']}, {MATERIAL_CONSTRAINTS['Flux_max']}]")
    
    reductant_qty = material_quantities.get('Reductant_qty', 0)
    if not (MATERIAL_CONSTRAINTS['Reductant_min'] <= reductant_qty <= MATERIAL_CONSTRAINTS['Reductant_max']):
        violations.append(f"Reductant quantity out of range: {reductant_qty:.0f} kg not in [{MATERIAL_CONSTRAINTS['Reductant_min']}, {MATERIAL_CONSTRAINTS['Reductant_max']}]")
    
    auxiliary_qty = material_quantities.get('Additive_qty', 0)
    if not (MATERIAL_CONSTRAINTS['Auxiliary_min'] <= auxiliary_qty <= MATERIAL_CONSTRAINTS['Auxiliary_max']):
        violations.append(f"Auxiliary quantity out of range: {auxiliary_qty:.2f} kg not in [{MATERIAL_CONSTRAINTS['Auxiliary_min']}, {MATERIAL_CONSTRAINTS['Auxiliary_max']}]")
    
    is_valid = len(violations) == 0
    return is_valid, violations

# ============================================================================
# COST CALCULATION
# ============================================================================

def calculate_cost(materials_dict: Dict[str, float],
                  material_master: pd.DataFrame,
                  sp_power_kwh: float,
                  power_rate: float = 6.50) -> Dict[str, float]:
    """
    Calculate total production cost.
    
    Formula:
    - Material_cost = sum(qty * cost_per_kg)
    - Power_cost = sp_power_kwh * power_rate
    - Total_cost = Material_cost + Power_cost
    
    Args:
        materials_dict: {material_name: quantity_kg}
        material_master: Material database with costs
        sp_power_kwh: Specific power consumption (kWh/MT)
        power_rate: Power cost (INR/kWh), default 6.50
    
    Returns:
        Dictionary with cost breakdown
    """
    material_cost = 0.0
    
    for material_name, qty in materials_dict.items():
        material = material_master[material_master['Material_Name'] == material_name].iloc[0]
        # Cost_per_kg is per MT (1000 kg) in material master
        cost_per_kg = material['Cost_per_kg'] / 1000.0
        material_cost += qty * cost_per_kg
    
    power_cost = sp_power_kwh * power_rate
    total_cost = material_cost + power_cost
    
    return {
        'material_cost': material_cost,
        'power_cost': power_cost,
        'total_cost': total_cost
    }

# ============================================================================
# SPECIFIC CONSUMPTION CALCULATIONS
# ============================================================================

def calculate_specific_consumption(materials_dict: Dict[str, float],
                                   material_master: pd.DataFrame,
                                   alloy_composition: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate specific consumption metrics.
    
    Formulas:
    - Sp_Cr_ore = Ore_kg / Cr_recovered
    - Sp_Carbon = FC_kg / Cr_recovered
    - FC_in_feed = (FC_kg / Total_Charge) * 100
    
    Args:
        materials_dict: {material_name: quantity_kg}
        material_master: Material database
        alloy_composition: Alloy composition data
    
    Returns:
        Dictionary with specific consumption metrics
    """
    total_ore = 0.0
    total_FC = 0.0
    total_charge = sum(materials_dict.values())
    
    for material_name, qty in materials_dict.items():
        material = material_master[material_master['Material_Name'] == material_name].iloc[0]
        
        if material['Material_Type'] == 'Ore':
            total_ore += qty
        
        total_FC += qty * (material['FC_%'] / 100.0)
    
    Cr_recovered = alloy_composition['Cr_recovered_kg']
    
    if Cr_recovered > 0:
        sp_cr_ore = total_ore / Cr_recovered
        sp_carbon = total_FC / Cr_recovered
    else:
        sp_cr_ore = 0.0
        sp_carbon = 0.0
    
    if total_charge > 0:
        fc_in_feed = (total_FC / total_charge) * 100
    else:
        fc_in_feed = 0.0
    
    return {
        'sp_cr_ore': sp_cr_ore,
        'sp_carbon': sp_carbon,
        'fc_in_feed': fc_in_feed,
        'total_ore_kg': total_ore,
        'total_fc_kg': total_FC
    }
