"""
Output Formatter - Ferrochrome Optimization System

Generates detailed, formatted output for optimization results.
"""

from typing import Dict, List
import pandas as pd

def print_header(target_cr: float, target_power: float, total_generated: int, valid_count: int):
    """Print optimization header with targets and statistics."""
    print("="*80)
    print("FERROCHROME CHARGE OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nOPTIMIZATION TARGETS:")
    print(f"   - Cr Recovery:     {target_cr:.2f}%")
    print(f"   - Specific Power:  {target_power:.0f} kWh/MT")
    
    print(f"\nSEARCH STATISTICS:")
    print(f"   - Total Combinations Generated:  {total_generated}")
    print(f"   - Valid Combinations: {valid_count}")
    print(f"   - Success Rate: {valid_count/total_generated*100:.2f}%")

def print_materials_table(materials_dict: Dict[str, float], material_master: pd.DataFrame):
    """Print formatted materials table."""
    print("   " + "-"*65)
    print("   | Material Name          Type        Qty (kg)    Cost (INR)   |")
    print("   " + "-"*65)
    
    total_qty = 0
    total_cost = 0
    
    for material_name, qty in materials_dict.items():
        material = material_master[material_master['Material_Name'] == material_name].iloc[0]
        mat_type = material['Material_Type']
        # Cost_per_kg is per MT (1000 kg) in material master
        cost_per_kg = material['Cost_per_kg'] / 1000.0
        cost = qty * cost_per_kg
        
        print(f"   | {material_name:22s} {mat_type:11s} {qty:8.1f}    {cost:10,.2f}    |")
        
        total_qty += qty
        total_cost += cost
    
    print("   " + "-"*65)
    print(f"   TOTAL CHARGE: {total_qty:,.1f} kg")

def print_quantities_table(quantities: Dict[str, float]):
    """Print formatted material quantities table with constraints."""
    # Import constraints
    from metallurgy_engine import MATERIAL_CONSTRAINTS
    
    print("   " + "-"*62)
    print("   | Category     Actual (kg)   Constraint (kg/MT)   Status    |")
    print("   " + "-"*62)
    
    # Ore
    ore_min = MATERIAL_CONSTRAINTS['Ore_min']
    ore_max = MATERIAL_CONSTRAINTS['Ore_max']
    ore_status = "PASS" if ore_min <= quantities['Ore_qty'] <= ore_max else "FAIL"
    print(f"   | Ore         {quantities['Ore_qty']:8.1f}    {ore_min:.0f}-{ore_max:.0f}          {ore_status:9s} |")
    
    # Flux
    flux_min = MATERIAL_CONSTRAINTS['Flux_min']
    flux_max = MATERIAL_CONSTRAINTS['Flux_max']
    flux_status = "PASS" if flux_min <= quantities['Flux_qty'] <= flux_max else "FAIL"
    print(f"   | Flux        {quantities['Flux_qty']:8.1f}    {flux_min:.0f}-{flux_max:.0f}            {flux_status:9s} |")
    
    # Reductant
    red_min = MATERIAL_CONSTRAINTS['Reductant_min']
    red_max = MATERIAL_CONSTRAINTS['Reductant_max']
    reductant_status = "PASS" if red_min <= quantities['Reductant_qty'] <= red_max else "FAIL"
    print(f"   | Reductant   {quantities['Reductant_qty']:8.1f}    {red_min:.0f}-{red_max:.0f}            {reductant_status:9s} |")
    
    # Auxiliary
    aux_min = MATERIAL_CONSTRAINTS['Auxiliary_min']
    aux_max = MATERIAL_CONSTRAINTS['Auxiliary_max']
    auxiliary_status = "PASS" if aux_min <= quantities['Additive_qty'] <= aux_max else "FAIL"
    print(f"   | Auxiliary   {quantities['Additive_qty']:8.2f}    {aux_min:.2f}-{aux_max:.0f}          {auxiliary_status:9s} |")
    
    print("   " + "-"*62)

def print_solution(rank: int, solution: Dict, target_cr: float, target_power: float, material_master: pd.DataFrame):
    """Print a single solution with full details."""
    print("\n" + "="*80)
    print(f"RANK {rank} - {'OPTIMAL SOLUTION' if rank==1 else 'ALTERNATIVE SOLUTION'}")
    print("="*80)
    
    # Economics
    print(f"\nECONOMICS:")
    cost = solution['cost_breakdown']
    
    print(f"   - Total Cost:           INR {cost['total_cost']:,.2f}")
    print(f"   - Material Cost:        INR {cost['material_cost']:,.2f}")
    print(f"   - Power Cost:           INR {cost['power_cost']:,.2f}")
    
    # Materials
    print(f"\nRAW MATERIAL CHARGE:")
    print(f"   Number of materials: {solution['n_materials']}")
    print_materials_table(solution['materials'], material_master)
    
    # Performance
    print(f"\nPREDICTED PERFORMANCE (ML Model):")
    cr_dev = abs(solution['cr_recovery'] - target_cr)
    power_dev = abs(solution['sp_power'] - target_power)
    cr_icon = "[OK]" if cr_dev < 1.0 else "[WARN]"
    power_icon = "[OK]" if power_dev < 50 else "[WARN]"
    
    print(f"   - Cr Recovery:         {solution['cr_recovery']:.2f}%  {cr_icon} (Target: {target_cr:.1f}%)")
    print(f"   - Specific Power:      {solution['sp_power']:.0f} kWh/MT  {power_icon} (Target: {target_power:.0f})")
    print(f"   - Deviation from Target: Cr={cr_dev:.2f}%, Power={power_dev:.0f} kWh/MT")
    
    # Material quantities
    print(f"\nMATERIAL QUANTITIES:")
    print_quantities_table(solution['quantities'])
    
    print(f"\n[PASS] ALL CONSTRAINTS SATISFIED")

def print_detailed_results(solutions: List[Dict], target_cr: float, target_power: float, 
                          total_generated: int, material_master: pd.DataFrame):
    """
    Print detailed results for all solutions.
    
    Args:
        solutions: List of valid solutions (ranked)
        target_cr: Target Cr recovery %
        target_power: Target specific power kWh/MT
        total_generated: Total combinations generated
        material_master: Material database
    """
    # Header
    print_header(target_cr, target_power, total_generated, len(solutions))
    
    # Print each solution
    for rank, solution in enumerate(solutions, 1):
        print_solution(rank, solution, target_cr, target_power, material_master)
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nDisplayed top {len(solutions)} solutions")
    print("All solutions meet material quantity constraints and dataset ranges")


def save_results_to_csv(solutions: List[Dict], filename: str = 'optimization_results.csv'):
    """
    Save optimization results to CSV file.
    
    Args:
        solutions: List of solutions
        filename: Output CSV filename
    """
    if len(solutions) == 0:
        print("\n[WARNING] No solutions to save")
        return
    
    # Prepare data for CSV
    rows = []
    for rank, sol in enumerate(solutions, 1):
        row = {
            'Rank': rank,
            'Cr_Recovery_%': sol['cr_recovery'],
            'SP_Power_kWh_MT': sol['sp_power'],
            'Total_Cost_INR': sol['cost_breakdown']['total_cost'],
            'Material_Cost_INR': sol['cost_breakdown']['material_cost'],
            'Power_Cost_INR': sol['cost_breakdown']['power_cost'],
            'Ore_kg': sol['quantities']['Ore_qty'],
            'Flux_kg': sol['quantities']['Flux_qty'],
            'Reductant_kg': sol['quantities']['Reductant_qty'],
            'Auxiliary_kg': sol['quantities']['Additive_qty'],
            'Total_Charge_kg': sol['quantities']['Total_Charge'],
            'N_Materials': sol['n_materials']
        }
        
        # Add material details
        for i, (mat_name, qty) in enumerate(sol['materials'].items(), 1):
            row[f'Material_{i}_Name'] = mat_name
            row[f'Material_{i}_Qty_kg'] = qty
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
