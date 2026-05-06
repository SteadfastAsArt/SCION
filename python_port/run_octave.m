%% Headless Octave runner for SCION baseline export.
%% Usage:  octave --no-gui --no-window-system --eval "run('python_port/run_octave.m')"
%% Run from the project root /home/laz/proj/SCION.

pkg load io                                  % needed for xlsread (must come BEFORE addpath -begin)

% Resolve project root from this file's location so the runner works regardless
% of pwd (Octave's `run` cd's into the script directory before executing).
this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);

% Make project SCION_*.m visible.  Order matters: stubs MUST stay in front of
% project root so SCION_plot_worldgraphic / SCION_plot_fluxes / xlsread / ode15s
% all resolve to our stubs first.  -begin prepends, so addpath the project
% root FIRST, then the stub dir LAST (which puts stubs at the very front).
addpath(project_root, '-begin');
addpath(fullfile(this_dir, 'octave_stubs'), '-begin');

% SCION_initialise expects relative paths like 'forcings/...'; cd to project root.
cd(project_root);

t_total = tic;
fprintf('=== SCION Octave baseline run ===\n');

% Octave looks up scripts in the CWD before searching the path, so the
% original SCION_plot_*.m files in project_root take precedence over the
% stubs unless we also chdir somewhere else. Wrapping in try/catch is
% simpler: SCION_initialise's failure happens only inside the plotting
% epilogue (after integration AND state assembly succeed), so the global
% `state`/`gridstate`/`pars` already contain the data we need to save.
global state gridstate pars forcings
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

out_path = fullfile(this_dir, 'scion_octave_baseline.mat');
% -v7 ensures scipy.io.loadmat compatibility (no HDF5).
save('-v7', out_path, 'state', 'gridstate', 'pars');
fprintf('Saved baseline -> %s\n', out_path);
