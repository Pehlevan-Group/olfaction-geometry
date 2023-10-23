function [ax ] = FormatAxis(varargin)
%Format axis as we like them 

default_args={...
'font',             15;......  
};
[g,error] = parse_args(default_args,varargin{:});

ax=gca;
ax.LineWidth=2;
ax.Box='off';
ax.TickDir='out';
ax.FontSize=g.font;
ax.XLabel.FontSize=g.font;
ax.YLabel.FontSize=g.font;
ax.FontName='Arial';
ax.XAxis.Color=[0 0 0];
ax.YAxis.Color=[0 0 0];

%   Detailed explanation goes here


end
