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

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1 : len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")



def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)

def _MTOW_init(problem):
    engine_fuel_type = problem.get_val("data:propulsion:fuel_type")
    NPAX_design = problem.get_val("data:TLAR:NPAX_design")
    NPAX = 2 + NPAX
    V_cruise = problem.get_val("data:TLAR:v_cruise")
    engine_count = problem.get_val("data:geometry:propulsion:engine:count")
    if engine_fuel_type < 3:
        MTOW = 363.804*NPAX_design + 5.9431*V_cruise
        return MTOW
    else:
        if engine_count < 2:
            MTOW = 557.1504*NPAX_design + 0.0157*V_cruise**2 + 0.2666*NPAX_design*V_cruise
            return MTOW
        else:
            MTOW = 635.6114*NPAX_design + 0.0229*V_cruise**2
            return MTOW


def residuals_analyzer(recorder_path):
    # Does not bring much info since the bloody reluctance is so high ...

    cr = om.CaseReader(recorder_path)

    solver_case = cr.get_cases("root.nonlinear_solver")
    variable_dict = {name: 0.0 for name in solver_case[-1].residuals}
    for case in solver_case:
        for residual in case.residuals:
            # Because those are matrix and I don't want to deal with it
            #if "aerodynamics:propeller" not in residual:
            variable_dict[residual] = variable_dict[residual] + np.sum(np.abs(case.residuals[residual]))
    sorted_variable_dict = dict(sorted(variable_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_variable_dict

                

      

def oad_process_vlm_sr22():
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
    
    # Removing previous case and adding a recorder
    recorder_path = pth.join(RESULTS_FOLDER_PATH, "sr22_cases.sql")


    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True

    problem.setup()
    problem.run_model()
    problem.write_outputs()
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

    

def oad_process_vlm_be76():
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
    # Removing previous case and adding a recorder
    recorder_path = pth.join(RESULTS_FOLDER_PATH, "be76_cases.sql")
    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True

    problem.setup()
    problem.run_model()
    problem.write_outputs()

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




def oad_process_tbm_900():
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
    # Removing previous case and adding a recorder
    recorder_path = pth.join(RESULTS_FOLDER_PATH, "tbm_900_cases.sql")
    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True
    problem.setup()
    problem.run_model()
    problem.write_outputs()

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


def oad_process_twin_otter_400():
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
    # Removing previous case and adding a recorder
    recorder_path = pth.join(RESULTS_FOLDER_PATH, "twin_otter_400_cases.sql")
    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True
    problem.setup()
    problem.run_model()
    problem.write_outputs()

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



oad_process_vlm_be76()



    




