function [num, txt, raw, lim] = xlsread(filename, varargin) %#ok<INUSD>
%% Octave-compatible xlsread shim (drops the MATLAB 'basic' interface flag).
%% SCION calls xlsread(file,'','','basic'); Octave's io xlsread rejects 'basic'.
%% This shim ignores all extra arguments and returns the numeric matrix
%% covering the used range of the first sheet — sufficient for SCION's two
%% xlsx forcings (GR_BA.xlsx, GA_revised.xlsx), each a single 2-column sheet.
    pkg load io;
    xls          = xlsopen(filename);
    [raw, ~, lim] = xls2oct(xls);
    xls          = xlsclose(xls); %#ok<NASGU>
    [num, txt]   = parsecell(raw, lim);
endfunction
