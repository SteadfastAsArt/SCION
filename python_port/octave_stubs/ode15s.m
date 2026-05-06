function [T, Y] = ode15s(odefun, tspan, y0, options)
%% MATLAB-style ode15s shim implemented on top of Octave's robust lsode.
%% Octave's bundled ode15s (SUNDIALS IDAS) fails the first stiff Newton on
%% SCION's spatial-weathering RHS (IDASolve corrector convergence error),
%% so for headless replication we fall back to lsode which integrates the
%% same BDF family of methods reliably.
%%
%% Behavioural contract preserved for SCION:
%%   - returns column vectors T (Nx1) and Y (Nxd) like MATLAB's ode15s
%%   - calls odefun(t, y) (note MATLAB time-first order)
%%   - T is the *actual sequence of t values passed to odefun*, not lsode's
%%     output-grid request — this is what SCION_initialise's
%%     intersect(workingstate.time, rawoutput.T, 'stable') depends on to
%%     reconstruct state vectors.  Returning lsode's output grid would
%%     leave intersect with no matches because lsode evaluates f at adaptive
%%     internal step points and dense-output-interpolates at requested grid.

    if nargin < 4 || isempty(options); options = struct(); end

    rtol = 1e-6; atol = 1e-9; max_step = Inf;
    if isstruct(options)
        if isfield(options, 'RelTol') && ~isempty(options.RelTol)
            rtol = options.RelTol;
        end
        if isfield(options, 'AbsTol') && ~isempty(options.AbsTol)
            atol = options.AbsTol;
        end
        if isfield(options, 'MaxStep') && ~isempty(options.MaxStep)
            max_step = options.MaxStep;
        end
    end

    lsode_options('relative tolerance', rtol);
    lsode_options('absolute tolerance', atol);
    lsode_options('integration method', 'stiff');
    if isfinite(max_step)
        lsode_options('maximum step size', max_step);
    end
    lsode_options('maximum order', 5);

    t0 = tspan(1); tf = tspan(end);
    if numel(tspan) > 2
        T_request = tspan(:);
    else
        npts = 5001;
        T_request = linspace(t0, tf, npts).';
    end

    if tf >= t0
        lo = t0; hi = tf;
    else
        lo = tf; hi = t0;
    end

    % Use a global recorder so the wrapped f can append (t, y) on every call.
    % Two globals are used to keep this isolated from any user globals.
    global ode15s_shim_t ode15s_shim_y
    ode15s_shim_t = zeros(0, 1);
    ode15s_shim_y = zeros(numel(y0), 0);

    f = @(y, t) ode15s_shim_record(t, y, odefun, lo, hi);

    lsode(f, y0(:), T_request);  % integrate; output values are discarded —
                                  % we use the recorded (t, y) sequence.

    T = ode15s_shim_t;
    Y = ode15s_shim_y.';

    clear global ode15s_shim_t ode15s_shim_y
endfunction


function ydot = ode15s_shim_record(t, y, odefun, lo, hi)
    global ode15s_shim_t ode15s_shim_y
    tc = max(lo, min(t, hi));
    ydot = odefun(tc, y);
    ode15s_shim_t(end+1, 1)    = tc;
    ode15s_shim_y(:, end+1)    = y(:);
endfunction
