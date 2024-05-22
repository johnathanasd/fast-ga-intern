"""Test module for Overall Aircraft Design process."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import os.path as pth
import logging
import shutil
from shutil import rmtree
from platform import system
import numpy as np
import openmdao.api as om
import csv
from numpy.testing import assert_allclose
import fastoad.api as oad
from fastga.models.performances.mission import resources
import time


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1 : len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")

_LOGGER = logging.getLogger(__name__)

def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)

def _MTOW_init(problem): # MTOW is in lbs
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    NPAX_design = problem.get_val("data:TLAR:NPAX_design")
    range = problem.get_val("data:TLAR:range",units = "NM")
    NPAX = 2 + NPAX_design
    V_cruise = problem.get_val("data:TLAR:v_cruise", units = "knot")
    V_cruise = 1.68780986 * V_cruise
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    if engine_fuel_type < 3:
        if engine_count <2:
            MTOW = 500*NPAX + 0.005*V_cruise*range
        else:
            MTOW = 530*NPAX + 0.02*V_cruise**2 + 0.2*NPAX*V_cruise
        return MTOW
    else:
        if engine_count < 2:
            MTOW = 530*NPAX + 0.012*V_cruise**2 + 0.2*NPAX*V_cruise
            return MTOW
        else:
            MTOW = 601.84*NPAX + 0.0081*V_cruise*range
            return MTOW

def _OWE_init(problem,MTOW): # OWE is in lbs
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    if engine_count < 2:
        OWE = 0.8841 - 0.0333*np.log(MTOW)
    else:
        if engine_fuel_type < 3:
            OWE = 0.4074 + 0.0253*np.log(MTOW)
        else:
            OWE = 0.5319 + 0.0066*np.log(MTOW)
    return OWE

def _MLW_init(problem,OWE):
    max_pl = problem.get_val("data:weight:aircraft:max_payload", units = "lbm")
    MLW_MZFW_ratio = problem.get_val("settings:weight:aircraft:MLW_MZFW_ratio")
    MLW = (max_pl+OWE)*MLW_MZFW_ratio
    return MLW

def _wing_area_init(problem,MTOW): # wing area is in ft^2
    V_app = problem.get_val("data:TLAR:v_approach", units = "knot")
    V_app = 1.68780986*V_app
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    if engine_fuel_type < 3:
        if engine_count < 2:
            wing_area = 130*MTOW/V_app**2 + 0.04*MTOW
        else:
            wing_area = 150 * MTOW/V_app**2 + 120
        return wing_area
    else:
        if engine_count < 2:
            wing_area = 135 * MTOW/V_app**2 + 150
            return wing_area
        else:
            wing_area = 121.7444 * MTOW/V_app**2 + 268.1264
            return wing_area

def _overall_aircraft_length_init(problem,MTOW):
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    NPAX_design = problem.get_val("data:TLAR:NPAX_design")
    NPAX = 2 + NPAX_design    
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    if engine_fuel_type < 3:
        overall_AC_length = 0.0116*MTOW - 0.0008*MTOW*NPAX
        return overall_AC_length
    else:
        if engine_count < 2:
            overall_AC_length = 0.0028 * MTOW + 2.056 * NPAX
            return overall_AC_length
        else:
            overall_AC_length = 0.5*np.sqrt(MTOW)
            return overall_AC_length

def _engine_cg_position_init(problem,MTOW,overall_AC_length): # engine cg position w.r.t the nose is in ft
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    if engine_fuel_type < 3:
        if engine_count < 2:
            engine_cg_pos =  0.0007*MTOW + 1.0445
        else:
            engine_cg_pos = 0.25*overall_AC_length
        return engine_cg_pos
    else:
        if engine_count < 2:
            engine_cg_pos = 0.00075 * MTOW + 1.589 
            return engine_cg_pos
        else:
            engine_cg_pos = 0.33*overall_AC_length
            return engine_cg_pos

def _wing_MAC_and_span_init(problem,wing_area): # both span and MAC qre in ft
    aspect_ratio = problem.get_val("data:geometry:wing:aspect_ratio")
    wing_span = np.sqrt(aspect_ratio*wing_area)
    wing_MAC = np.sqrt(wing_area/aspect_ratio)
    return wing_span,wing_MAC

def _control_surface_position_init(problem,overall_AC_length): # all distance w.r.t the nose are in ft
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    T_tail = problem.get_val("data:geometry:has_T_tail")
    if engine_fuel_type < 3:
        if engine_count < 2:
            wing_25 = 0.3132 * overall_AC_length
        else:
            wing_25 = 0.351 * overall_AC_length
        wing_25_cg_diff = 0.0514 * overall_AC_length
        wing_cg = wing_25 + wing_25_cg_diff
        if T_tail < 1:
            htp_cg = 0.91 * overall_AC_length
            vtp_cg = 0.891 * overall_AC_length
        else:
            htp_cg = 0.96 * overall_AC_length
            vtp_cg = 0.93 * overall_AC_length
        return wing_25, wing_cg, htp_cg, vtp_cg
    else:
        if engine_count < 2:
            wing_25 = 0.351 * overall_AC_length
            wing_25_cg_diff = 0.0494 * overall_AC_length
            wing_cg = wing_25 + wing_25_cg_diff
            if T_tail < 1:
                htp_cg = 0.92 * overall_AC_length
                vtp_cg = 0.895 * overall_AC_length
            else:
                htp_cg = 0.92 * overall_AC_length
                vtp_cg = 0.85 * overall_AC_length
            return wing_25, wing_cg, htp_cg, vtp_cg
        else:
            wing_25 = 0.379 * overall_AC_length
            wing_25_cg_diff = 0.038 * overall_AC_length
            wing_cg = wing_25 + wing_25_cg_diff
            if T_tail < 1:
                htp_cg = 0.906 * overall_AC_length
                vtp_cg = 0.899 * overall_AC_length
            else:
                htp_cg = 0.94 * overall_AC_length
                vtp_cg = 0.865 * overall_AC_length
            return wing_25, wing_cg, htp_cg, vtp_cg
        
def _htp_vtp_area_init(problem,wing_span,wing_MAC,wing_cg,htp_cg,vtp_cg,wing_area): # both areas are in ft^2
    L_cg_htp = htp_cg - wing_cg
    L_cg_vtp = vtp_cg - wing_cg
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    if engine_fuel_type < 3:
        if engine_count < 2:
            htp_area = 0.672 * wing_MAC*wing_area/L_cg_htp
            vtp_area = 0.0443 *wing_span*wing_area/L_cg_vtp
        else:
            htp_area = 0.812 * wing_MAC*wing_area/L_cg_htp
            vtp_area = 0.0657 * wing_span*wing_area/L_cg_vtp
        return htp_area, vtp_area
    else:
        if engine_count < 2:
            htp_area = 1.75*0.672 * wing_MAC*wing_area/L_cg_htp
            vtp_area = 1.75*0.0443 *  wing_span*wing_area/L_cg_vtp
        else:
            htp_area = 0.930 * wing_MAC*wing_area/L_cg_htp
            vtp_area = 0.0707 * wing_span*wing_area/L_cg_vtp
        return htp_area, vtp_area

def _atmosphere(altitude): # altitude in meters   
    T = 288.15 - 0.0065*altitude # Temperature in K
    Ps= 101325*(1-altitude/44330.7792)**5.25587611 # Static pressure [Pa]
    rho = Ps/T/287.051124 # Air density in kg/m^3 = 0.0624279606 lb/ft^3
    return rho

def _wing_weight(problem,wing_area,MTOW,OWE):
    altitude = problem.get_val("data:mission:sizing:main_route:cruise:altitude",units="m")
    rho = _atmosphere(altitude)
    rho = 0.0624279606*rho
    V_cruise = problem.get_val("data:TLAR:v_cruise", units = "knot")
    V_cruise = 1.68780986 * V_cruise # ft/s
    q = 0.5*rho*V_cruise**2 # lb/ft^2
    AR_w = problem.get_val("data:geometry:wing:aspect_ratio")
    Nz = problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft")
    t_c = problem.get_val("data:geometry:wing:thickness_ratio")
    wing_TR = problem.get_val("data:geometry:wing:taper_ratio")
    wing_sweep_25 = problem.get_val("data:geometry:wing:sweep_25",units = "rad")
    fw = MTOW - problem.get_val("data:weight:aircraft:payload",units = "lbm") - OWE
    frac1 = AR_w/(np.cos(wing_sweep_25)**2)
    frac2 = 100*t_c/np.cos(wing_sweep_25)
    wing_weight = 0.036*pow(wing_area,0.758)*pow(fw,0.0035)*pow(frac1,0.6)*pow(q,0.006)*pow(wing_TR,0.04)*pow(frac2,-0.3)*pow(Nz*MTOW,0.49)
    return wing_weight

def _htp_weight(problem,htp_area,MTOW):
    altitude = problem.get_val("data:mission:sizing:main_route:cruise:altitude",units="m")
    rho = _atmosphere(altitude)
    rho = 0.0624279606*rho
    V_cruise = problem.get_val("data:TLAR:v_cruise", units = "knot")
    V_cruise = 1.68780986 * V_cruise # ft/s
    Nz = problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft")
    q = 0.5*rho*V_cruise**2 # lb/ft^2
    htp_sweep_25 = problem.get_val("data:geometry:horizontal_tail:sweep_25",units="rad")
    htp_TR = problem.get_val("data:geometry:horizontal_tail:taper_ratio")
    AR_htp = problem.get_val("data:geometry:horizontal_tail:aspect_ratio")
    t_c = problem.get_val("data:geometry:wing:thickness_ratio")
    frac1 = 100*t_c/np.cos(htp_sweep_25)
    frac2 = AR_htp/np.cos(htp_sweep_25)**2
    htp_weight = 0.016*pow(Nz*MTOW,0.414)*pow(q,0.168)*pow(htp_area,0.896)*pow(frac1,-0.12)*pow(frac2,0.043)*pow(htp_TR,-0.02)
    return htp_weight

def _vtp_weight(problem,vtp_area,MTOW):
    altitude = problem.get_val("data:mission:sizing:main_route:cruise:altitude",units="m")
    rho = _atmosphere(altitude)
    rho = 0.0624279606*rho
    V_cruise = problem.get_val("data:TLAR:v_cruise", units = "knot")
    V_cruise = 1.68780986 * V_cruise # ft/s
    Nz = problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft")
    q = 0.5*rho*V_cruise**2 # lb/ft^2
    has_T_tail = problem.get_val("data:geometry:has_T_tail")
    vtp_sweep_25 = problem.get_val("data:geometry:vertical_tail:sweep_25",units="rad")
    vtp_TR = problem.get_val("data:geometry:vertical_tail:taper_ratio")
    AR_vtp = problem.get_val("data:geometry:vertical_tail:aspect_ratio")
    t_c = problem.get_val("data:geometry:wing:thickness_ratio")
    frac1 = 100*t_c/np.cos(vtp_sweep_25)
    frac2 = AR_vtp/np.cos(vtp_sweep_25)**2
    vtp_weight = 0.073*(1+0.2*has_T_tail)*pow(Nz*MTOW,0.376)*pow(q,0.122)*pow(vtp_area,0.873)*pow(frac1,-0.49)*pow(frac2,0.357)*pow(vtp_TR,0.039)
    return vtp_weight

def loop_initialization(problem):
    MTOW = _MTOW_init(problem)
    wing_area = _wing_area_init(problem,MTOW)
    overall_AC_length = _overall_aircraft_length_init(problem,MTOW)
    engine_cg_pos = _engine_cg_position_init(problem,MTOW,overall_AC_length)
    wing_span,wing_MAC = _wing_MAC_and_span_init(problem,wing_area)
    wing_25, wing_cg, htp_cg, vtp_cg = _control_surface_position_init(problem,overall_AC_length)
    htp_area, vtp_area = _htp_vtp_area_init(problem,wing_span,wing_MAC,wing_cg,htp_cg,vtp_cg,wing_area)
    OWE = _OWE_init(problem,MTOW)
    MLW = _MLW_init(problem,OWE)
    wing_weight = _wing_weight(problem,wing_area,MTOW,OWE)
    htp_weight = _htp_weight(problem,htp_area,MTOW)
    vtp_weight = _vtp_weight(problem,vtp_area,MTOW)
    # set value
    problem.set_val("data:weight:aircraft:MTOW", val=MTOW, units="lbm")
    problem.set_val("data:geometry:wing:area", val=wing_area, units="ft**2")
    problem.set_val("data:geometry:horizontal_tail:area", val=htp_area, units="ft**2")
    problem.set_val("data:geometry:vertical_tail:area", val=vtp_area, units="ft**2")
    problem.set_val("data:geometry:wing:MAC:length", val=wing_MAC, units="ft")
    problem.set_val("data:geometry:wing:MAC:at25percent:x", val=wing_25, units="ft")
    problem.set_val("data:weight:airframe:wing:CG:x", val=wing_cg, units="ft")
    problem.set_val("data:weight:airframe:horizontal_tail:CG:x", val=htp_cg, units="ft")
    problem.set_val("data:weight:airframe:vertical_tail:CG:x", val=vtp_cg, units="ft")
    problem.set_val("data:weight:propulsion:engine:CG:x", val=engine_cg_pos, units="ft")
    problem.set_val("data:weight:aircraft:OWE", val=OWE, units="lbm")
    problem.set_val("data:weight:aircraft:MLW", val=MLW, units="lbm")
    problem.set_val("data:weight:airframe:wing:mass", val=wing_weight, units="lbm")
    problem.set_val("data:weight:airframe:horizontal_tail:mass", val=htp_weight, units="lbm")
    problem.set_val("data:weight:airframe:vertical_tail:mass", val=vtp_weight, units="lbm")

    return problem

def value_comparison(problem):
    # get value
    MTOW = problem.get_val("data:weight:aircraft:MTOW", units="kg")
    wing_area = problem.get_val("data:geometry:wing:area", units="m**2")
    htp_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    vtp_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    wing_MAC = problem.get_val("data:geometry:wing:MAC:length", units="m")
    wing_25 = problem.get_val("data:geometry:wing:MAC:at25percent:x", units="m")
    wing_cg = problem.get_val("data:weight:airframe:wing:CG:x", units="m")
    htp_cg = problem.get_val("data:weight:airframe:horizontal_tail:CG:x", units="m")
    vtp_cg = problem.get_val("data:weight:airframe:vertical_tail:CG:x", units="m")
    engine_cg = problem.get_val("data:weight:propulsion:engine:CG:x", units="m")
    inp = np.array([MTOW,
                    wing_area,
                    htp_area,
                    vtp_area,
                    wing_MAC,
                    wing_25,
                    wing_cg,
                    htp_cg,
                    vtp_cg,
                    engine_cg])

    variable_names = ["MTOW (kg)",
                      "Wing Area (m^2)",
                      "HTP Area (m^2)",
                      "VTP Area (m^2)",
                      "Wing MAC (m)",
                      "Wing 25% MAC (m)",
                      "Wing CG (m)",
                      "HTP CG (m)",
                      "VTP CG (m)",
                      "Engine CG (m)"]

    return inp, variable_names

def residuals_analyzer(recorder_path):
    # Does not bring much info since the bloody reluctance is so high ...

    cr = om.CaseReader(recorder_path)

    solver_case = cr.get_cases("root.nonlinear_solver")
    variable_dict = {name: 0.0 for name in solver_case[-1].residuals}
    for case in solver_case:
        for residual in case.residuals:
            # Because those are matrix and I don't want to deal with it
            #if "aerodynamics:propeller" not in residual:
            #variable_dict[residual] = variable_dict[residual] + np.sum(np.abs(case.residuals[residual]))
            variable_dict[residual] = np.sum(np.abs(case.residuals[residual]))

    sorted_variable_dict = dict(sorted(variable_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_variable_dict

                

      

def oad_process_vlm_sr22(loop_init = False,recorder_opt = False,input_compare = False):
    cleanup()
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "oad_process_sr22.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)
    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    if recorder_opt is True:
    # Removing previous case and adding a recorder
        recorder_path = pth.join(RESULTS_FOLDER_PATH, "sr22_cases.sql")
        recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    if recorder_opt is True:
        solver.add_recorder(recorder)
        solver.recording_options["record_solver_residuals"] = True

    problem.setup()

    if loop_init is True:
        problem = loop_initialization(problem)
    
    if input_compare is True:
        input_init,variable_names = value_comparison(problem)    

    problem.run_model()
    problem.write_outputs()
    if input_compare is True:
        input_final,_ = value_comparison(problem)
        output = np.column_stack((np.array(variable_names).reshape(-1, 1),input_init.astype(str),input_final.astype(str)))
        return output

    if recorder_opt is True:
        sorted_variable_residuals = residuals_analyzer(recorder_path)
    

        # Create the folder if it doesn't exist
        os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

        # Construct the file path
        file_path = os.path.join(RESULTS_FOLDER_PATH, 'sr22_residuals_analysis.csv')

        # Open the file for writing
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(['Variable name', 'Sum of squared Residuals'])

            # Write the sum of residuals for each iteration
            for name, sum_res in sorted_variable_residuals.items():
                writer.writerow([name, sum_res])
    
    

def oad_process_vlm_be76(loop_init = False,recorder_opt = False,input_compare = False):
    cleanup()
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_be76.xml"
    process_file_name = "oad_process_be76.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    if recorder_opt is True:
        # Removing previous case and adding a recorder
        recorder_path = pth.join(RESULTS_FOLDER_PATH, "be76_cases.sql")
        recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    if recorder_opt is True:
        solver.add_recorder(recorder)
        solver.recording_options["record_solver_residuals"] = True

    problem.setup()

    if loop_init is True:
        problem = loop_initialization(problem)

    if input_compare is True:
        input_init,variable_names = value_comparison(problem)    

    problem.run_model()
    problem.write_outputs()
    if input_compare is True:
        input_final,_ = value_comparison(problem)
        output = np.column_stack((np.array(variable_names).reshape(-1, 1),input_init.astype(str),input_final.astype(str)))
        return output

    if recorder_opt is True:
        sorted_variable_residuals = residuals_analyzer(recorder_path)
    

        # Create the folder if it doesn't exist
        os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

        # Construct the file path
        file_path = os.path.join(RESULTS_FOLDER_PATH, 'be76_residuals_analysis.csv')

        # Open the file for writing
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(['Variable name', 'Sum of squared Residuals'])

            # Write the sum of residuals for each iteration
            for name, sum_res in sorted_variable_residuals.items():
                writer.writerow([name, sum_res])




def oad_process_tbm_900(loop_init = False,recorder_opt = False,input_compare = False):
    cleanup()
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_tbm900.xml"
    process_file_name = "oad_process_tbm900.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    if recorder_opt is True:
        # Removing previous case and adding a recorder
        recorder_path = pth.join(RESULTS_FOLDER_PATH, "tbm_900_cases.sql")
        recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    if recorder_opt is True:
        solver.add_recorder(recorder)
        solver.recording_options["record_solver_residuals"] = True
    problem.setup()
    if loop_init is True:
        problem = loop_initialization(problem)
    
    if input_compare is True:
        input_init,variable_names = value_comparison(problem)    

    problem.run_model()
    problem.write_outputs()
    if input_compare is True:
        input_final,_ = value_comparison(problem)
        output = np.column_stack((np.array(variable_names).reshape(-1, 1),input_init.astype(str),input_final.astype(str)))
        return output
    

    if recorder_opt is True:
        sorted_variable_residuals = residuals_analyzer(recorder_path)
    

        # Create the folder if it doesn't exist
        os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

        # Construct the file path
        file_path = os.path.join(RESULTS_FOLDER_PATH, 'tbm_900_residuals_analysis.csv')

        # Open the file for writing
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(['Variable name', 'Sum of squared Residuals'])

            # Write the sum of residuals for each iteration
            for name, sum_res in sorted_variable_residuals.items():
                writer.writerow([name, sum_res])


def oad_process_twin_otter_400(loop_init = False, recorder_opt = False,input_compare = False):
    cleanup()
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "Source File Twin Otter DHC6-400.xml"
    process_file_name = "oad_process_twin_otter_400.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_source = pth.join(DATA_FOLDER_PATH, xml_file_name)
    config_file = pth.join(DATA_FOLDER_PATH, process_file_name)
    ref_inputs = oad.generate_inputs(config_file, ref_source, overwrite=True)
    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    if recorder_opt is True:
        # Removing previous case and adding a recorder
        recorder_path = pth.join(RESULTS_FOLDER_PATH, "twin_otter_400_cases.sql")
        recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    if recorder_opt is True:
        solver.add_recorder(recorder)
        solver.recording_options["record_solver_residuals"] = True
    problem.setup()

    if loop_init is True:
        problem = loop_initialization(problem)

    if input_compare is True:
        input_init,variable_names = value_comparison(problem)    

    problem.run_model()
    problem.write_outputs()
    if input_compare is True:
        input_final,_ = value_comparison(problem)
        output = np.column_stack((np.array(variable_names).reshape(-1, 1),input_init.astype(str),input_final.astype(str)))
        return output

    if recorder_opt is True:
        sorted_variable_residuals = residuals_analyzer(recorder_path)
    

        # Create the folder if it doesn't exist
        os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

        # Construct the file path
        file_path = os.path.join(RESULTS_FOLDER_PATH, 'twin_otter_400_residuals_analysis.csv')

        # Open the file for writing
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(['Variable name', 'Sum of squared Residuals'])

            # Write the sum of residuals for each iteration
            for name, sum_res in sorted_variable_residuals.items():
                writer.writerow([name, sum_res])



loop_init = True

#output = oad_process_vlm_be76(loop_init = True,recorder_opt = False,input_compare = True)
#print(output)


loop_init = True
start_time = time.time()
oad_process_vlm_sr22(loop_init)
end_time = time.time()
sr22_init = end_time - start_time
print(sr22_init)

start_time = time.time()
oad_process_vlm_sr22()
end_time = time.time()
sr22 = end_time - start_time

start_time = time.time()
oad_process_vlm_be76(loop_init)
end_time = time.time()
be76_init = end_time - start_time

start_time = time.time()
oad_process_vlm_be76()
end_time = time.time()
be76 = end_time - start_time

start_time = time.time()
oad_process_tbm_900(loop_init)
end_time = time.time()
tbm_900_init = end_time - start_time

start_time = time.time()
oad_process_tbm_900()
end_time = time.time()
tbm_900 = end_time - start_time

start_time = time.time()
oad_process_twin_otter_400(loop_init)
end_time = time.time()
twin_otter_init = end_time - start_time

start_time = time.time()
oad_process_twin_otter_400()
end_time = time.time()
twin_otter = end_time - start_time


print("For TBM 900 the running time with initialization is {:.6f} sec, without is {:.6f}".format(tbm_900_init,tbm_900))
print("For Twin Otter the running time with initialization is {:.6f} sec, without is {:.6f}".format(twin_otter_init,twin_otter))
print("For SR22 the running time with initialization is {:.6f} sec, without is {:.6f}".format(sr22_init, sr22))
print("For BE76 the running time with initialization is {:.6f} sec, without is {:.6f}".format(be76_init, be76))


