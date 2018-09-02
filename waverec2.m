function x = waverec2(c,s,varargin)

if errargn(mfilename,nargin,[3:4],nargout,[0:1]), error('*'), end

x = appcoef2(c,s,varargin{:},0);
