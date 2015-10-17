%% This function implements the dynamic hiden algorithm.
% G: graph before modification.
% Gp: graph after modification.
% A: assignment for G
% K: number of vertices allowd to change hierarchy
% assignment: hierarchy for Gp
% cost: the cost for [Gp,assignment]

% Implemented by Dihong in June 06, 2014.

function [assignment,cost] = dynamic_hiden(G,Gp,A,K,M)
ranking = estimate_dynamic_set(G,Gp,A,1e-4);
if nargin==4
	[assignment,cost] = local_hiden(Gp,A,ranking(1:K),size(G,1));
else
	[assignment,cost] = local_hiden(Gp,A,ranking(1:K),M);
end
end
