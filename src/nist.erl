%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    Read nist data
%%% @end
%%% Created : 15 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(nist).

-export([open_image_file/0, open_image_file/1]).
-export([close_image_file/1]).
-export([read_next_image/1]).
-export([read_image/2]).
-export([read_image_vector/2]).

-export([open_label_file/0, open_label_file/1]).
-export([close_label_file/1]).
-export([read_next_label/1]).
-export([read_label/2]).
-export([read_label_number/2]).
-export([read_label_vector/2]).
-export([image_to_vector/1]).
-export([image_to_matrix/1]).
-export([label_to_vector/1]).
-export([label_to_matrix/1]).

-record(nist_image_file,
	{
	  fd,
	  n,
	  rows,
	  columns
	}).


-record(nist_label_file,
	{
	  fd,
	  n
	}).

open_image_file() ->
    open_image_file(filename:join(code:priv_dir(deep),
				  "train-images-idx3-ubyte.gz")).

open_image_file(File) ->
    {ok,Fd} = file:open(File, [read, compressed, binary]),
    case file:read(Fd, 4*4) of
	{ok, <<16#00000803:32, N:32, Rows:32, Columns:32>>} ->
	    {ok, #nist_image_file { fd=Fd, n=N, rows=Rows, columns=Columns}};
	{ok,_} ->
	    {error, bad_magic};
	E ={error,_Error} ->
	    E
    end.

close_image_file(#nist_image_file { fd=Fd }) ->
    file:close(Fd).

read_next_image(F) ->
    N = F#nist_image_file.rows*F#nist_image_file.columns,
    file:read(F#nist_image_file.fd, N).

%% read image by index, zero based
read_image(F, I) when I >= 0 ->
    N = F#nist_image_file.rows*F#nist_image_file.columns,
    Pos = 4*4 + I*N,
    file:position(F#nist_image_file.fd, Pos),
    file:read(F#nist_image_file.fd, N).

%% image as column vector (coded as matrix)
image_to_vector(Bin) when is_binary(Bin) ->
    matrix:from_list([[ X/255] || <<X>> <= Bin ],float32).

%% return the image as a column vector
image_to_matrix(Bin) when is_binary(Bin) ->
    matrix:from_list([ [ X/255 ] || <<X>> <= Bin ],float32).

label_to_vector(N) when is_integer(N), N>=0, N=< 9 ->
    matrix:from_list([[I] || 
			 I <- tuple_to_list(erlang:make_tuple(10,0.0,
							      [{N+1,1.0}]))],
		     float32).

label_to_matrix(N) ->
    label_to_vector(N).


%% read input data vector normalize in [0, 1]
read_image_vector(F, I) ->
    case read_image(F, I) of
	{ok, Bin} -> {ok, image_to_vector(Bin)};
	Error -> Error
    end.

open_label_file() ->
    open_label_file(filename:join(code:priv_dir(deep),
				  "train-labels-idx1-ubyte.gz")).

open_label_file(File) ->
    {ok,Fd} = file:open(File, [read, compressed, binary]),
    case file:read(Fd, 2*4) of
	{ok, <<16#00000801:32, N:32>>} ->
	    {ok, #nist_label_file { fd=Fd, n=N }};
	{ok,_} ->
	    {error, bad_magic};
	E ={error,_Error} ->
	    E
    end.

close_label_file(#nist_label_file { fd=Fd }) ->
    file:close(Fd).


read_next_label(F) ->
    file:read(F#nist_label_file.fd, 1).

%% read image by index, zero based
read_label(F, I) when I >= 0 ->
    Pos = 2*4 + I,
    file:position(F#nist_label_file.fd, Pos),
    file:read(F#nist_label_file.fd, 1).

read_label_number(F, I) ->
    case read_label(F, I) of
	{ok, <<N>>} ->
	    {ok,N};
	Error -> Error 
    end.

read_label_vector(F, I) ->
    case read_label_number(F, I) of
	{ok,N} -> {ok, label_to_vector(N)};
	Error -> Error
    end.
