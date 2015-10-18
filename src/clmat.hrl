-ifndef(__CLMAT_HRL__).
-define(__CLMAT_HRL__, true).

-include_lib("cl/include/cl.hrl").

-type unsigned() :: non_neg_integer().

-record(matrix,
	{
	  n :: unsigned(),
	  m :: unsigned(),
	  endian :: little | big,
	  type :: float | double,
	  data :: binary()
	}).

%% representation of matrix as openCL buffer object (by index i)
-record(clmat,
	{
	  i    :: unsigned(),  %% index to memory object
	  n    :: unsigned(),  %% rows
	  m    :: unsigned(),  %% columns
	  e    :: unsigned()   %% element size
	}).

-endif.
