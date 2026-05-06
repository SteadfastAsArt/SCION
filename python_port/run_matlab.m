%% MATLAB runner for SCION baseline export — used to generate ground truth
%% for `python_port/compare.py --matlab-mat ...`.
%%
%% Usage on a machine with MATLAB installed:
%%     1. cd to the SCION project root (the directory containing forcings/)
%%     2. Open MATLAB, then:
%%             >> run('python_port/run_matlab.m')
%%        — or from a shell:
%%             matlab -batch "run('python_port/run_matlab.m')"
%%     3. The output `python_port/scion_matlab_baseline.mat` will be written
%%        with the same `state` / `gridstate` / `pars` schema as the Octave
%%        baseline so compare.py can swap one for the other transparently.
%%
%% This script does NOT modify any of the upstream SCION_*.m files. It just
%% calls SCION_initialise(0) (full deterministic run) and saves the resulting
%% globals.

this_dir     = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);

% Make sure project SCION_*.m + helper scripts (interp1qr, tight_subplot) are
% on the path even if MATLAB was launched from elsewhere.
addpath(project_root);

% Run from the project root so SCION_initialise's relative `forcings/*.mat`
% and `forcings/*.xlsx` loads resolve.
old_pwd = pwd;
cd(project_root);
cleanup_pwd = onCleanup(@() cd(old_pwd));

fprintf('=== SCION MATLAB baseline run ===\n');
t_total = tic;

% SCION_initialise(0) runs full deterministic integration AND triggers the
% world-graphic + flux plotting epilogue. Plotting requires M_Map and
% topotoolbox's ttcmap. If those are not installed we still want to recover
% the integration output, so wrap in try/catch and pull from globals.
global state gridstate pars %#ok<GVMIS>
try
    result = SCION_initialise(0); %#ok<NASGU>
    fprintf('SCION_initialise returned cleanly.\n');
catch err
    fprintf('SCION_initialise raised after integration (likely plotting): %s\n', err.message);
    if isempty(state) || isempty(gridstate)
        rethrow(err);  % integration itself failed -> nothing to save
    end
    fprintf('Recovering state/gridstate/pars from globals (integration succeeded).\n');
end

elapsed = toc(t_total);
fprintf('Total wall time: %.1f s\n', elapsed);

out_path = fullfile(this_dir, 'scion_matlab_baseline.mat');
save(out_path, 'state', 'gridstate', 'pars', '-v7');
fprintf('Saved MATLAB baseline -> %s\n', out_path);
fprintf('Now run from the project root:\n');
fprintf('    python3 python_port/compare.py --matlab-mat python_port/scion_matlab_baseline.mat\n');
