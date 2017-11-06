%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    deep_net
%%% @end
%%% Created : 14 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(deep_net).

-export([new/3]).
-export([feed/2]).
-export([backprop/3]).
-export([learn/4]).
-export([sgd/4]).

-compile(export_all).

-record(layer,
	{
	  inputs  :: integer(),         %% number of inputs
	  outputs  :: integer(),        %% number of outputs
	  weights :: matrix:matrix(),          %% weight matrix
	  bias    :: matrix:matrix(),          %% bias vector
	  locked = false :: boolean()   %% update weights?
	}).


%% create a network with an input vector of size Ni 
%% then a list of sizes of the hidden layers, lastly the
%% output vector.
%%
new(IN, Hs, OUT) when is_integer(IN), IN > 0,
		      is_list(Hs),
		      is_integer(OUT), OUT > 0 ->
    create__(IN, Hs++[OUT]).

create__(IN,[OUT|Hs]) ->
    W = matrix:normal(OUT,IN,float32),  %% weights hidden layers
    B = matrix:normal(OUT,1,float32),   %% bias vector
    L = #layer { inputs = IN, outputs = OUT, weights = W, bias = B },
    [L | create__(OUT,Hs)];
create__(_IN,[]) ->
    [].

%%
%% Feed forward 
%%
feed(Net,IN) ->
    feed_(Net,IN).

feed_([#layer{weights=W,bias=B}|Net],A) ->
    Z = matrix:add(matrix:multiply(W,A), B),
    feed_(Net, matrix:sigmoid(Z));
feed_([], A) ->
    A.

sgd(Net,TrainingSet,ValidationSet,InputOptions)
      when is_list(TrainingSet),
	   is_list(ValidationSet),
	   is_map(InputOptions) ->
    Default = #{ eta => 3.0, 
		 lmbda => 0.0,
		 batch_size => 10,
		 epochs => 20 },
    Options = maps:merge(Default, InputOptions),
    Eta   = option(learning_rate, Options),
    Lmbda = option(regularization, Options),
    BatchSize = option(batch_size, Options),
    Epochs = option(epochs, Options),
    N = length(TrainingSet),
    sgd_(Net,N,TrainingSet,ValidationSet,1,Epochs,BatchSize,Eta,Lmbda).

sgd_(Net,_N,_TrainingSet,_ValidationSet,E,Epochs,_BatchSize,_Eta,_Lmbda)
  when E > Epochs -> Net;
sgd_(Net,N,TrainingSet,ValidationSet,E,Epochs,BatchSize,Eta,Lmbda) ->
    TrainingSet1 = deep_random:shuffle(TrainingSet),
    T0 = erlang:monotonic_time(),
    Net1 = sgd__(Net,TrainingSet1,N,BatchSize,Eta,Lmbda/N),
    T1 = erlang:monotonic_time(),
    R = evaluate(Net1,ValidationSet),
    T2 = erlang:monotonic_time(),
    M = length(ValidationSet),
    Time1 = erlang:convert_time_unit(T1 - T0, native, microsecond),
    Time2 = erlang:convert_time_unit(T2 - T1, native, microsecond),
    io:format("epoch ~w/~w : learn=~.2f%, train=~.2fs, eval=~.2fs\n", 
	      [E,Epochs,(R/M)*100,Time1/1000000,Time2/1000000]),
    sgd_(Net1,N,TrainingSet,ValidationSet,E+1,Epochs,BatchSize,Eta,Lmbda).

%% loop over batches and train the net
sgd__(Net, TrainingSet, N, BatchSize, Eta, Lmbda) when N >= BatchSize ->
    {Batch,TrainingSet1} = lists:split(BatchSize, TrainingSet),
    %% io:format("learn\n"),
    L = length(Batch),
    Net1 = learn(Net, Batch, Eta/L, Lmbda),
    sgd__(Net1, TrainingSet1, N-BatchSize, BatchSize, Eta, Lmbda);
sgd__(Net, _TrainingSet, _N, _BatchSize, _Eta, _Lmbda) ->
    Net.

%%
%% Run learning over a batch of {X,Y} pairs
%%
learn(Net, [{X,Y}|Batch], Eta, Lmbda) ->
    {DNw,DNb} = backprop(Net, X, Y),
    learn(Net, Batch, Eta, Lmbda, DNw, DNb).

learn(Net, [{X,Y}|Batch], Eta, Lmbda, Nw, Nb) ->
    {DNw,DNb} = backprop(Net, X, Y),
    Nw2 = add_lists(Nw, DNw),
    Nb2 = add_lists(Nb, DNb),
    learn(Net, Batch, Eta, Lmbda, Nw2, Nb2);
learn(Net, [], Eta, Lmbda, Nw, Nb) ->
    learn_net(Net, Nw, Nb, Eta, Lmbda).

learn_net([L=#layer{locked=true}|Net], [_NW|Ws], [_NB|Bs], Eta, Lmbda) ->
    [L | learn_net(Net, Ws, Bs, Eta, Lmbda)];
learn_net([L=#layer{weights=W,bias=B}|Net], [NW|Ws], [NB|Bs], Eta, Lmbda) ->
    W1 = matrix:subtract(matrix:scale((1-Eta*Lmbda),W), matrix:scale(Eta, NW)),
    B1 = matrix:subtract(B, matrix:scale(Eta, NB)),
    [L#layer{weights=W1,bias=B1} | learn_net(Net, Ws, Bs, Eta, Lmbda)];
learn_net([], [], [], _Eta, _Lmbda) ->
    [].

add_lists([A|As],[B|Bs]) ->
    add_lists_(As, Bs, [matrix:add(A,B)]).

add_lists_([A|As],[B|Bs],Cs) ->
    add_lists_(As,Bs,[matrix:add(A,B)|Cs]);
add_lists_([],[],Cs) ->
    lists:reverse(Cs).

%%
%% back propagation algorithm,
%% adjust weights and biases to match output Y on input X
%%
backprop(Net, X, Y) ->
    {[A1,A2|As],[Z|Zs]} = feed_forward_(Net, X, [X], []),
    SP = matrix:sigmoid_prime(Z),
    D0 = matrix:times(cost_derivative(A1, Y), SP),
    %% select ktop from cost_derivative => D0
    NetR = lists:reverse(tl(Net)),
    backward_pass_(NetR,D0,As,Zs,[matrix:multiply(D0,matrix:transpose(A2))],[D0]).

backward_pass_([#layer{weights=W,bias=_B}|Net],D,[A|As],[Z|Zs],Nw,Nb) ->
    SP = matrix:sigmoid_prime(Z),
    Wt = matrix:transpose(W),
    %% Dc = column order of D
    Dc   = matrix:transpose(matrix:transpose_data(D)), %% row->column order
    WtD  = matrix:multiply(Wt, Dc),
    D1 = matrix:times(WtD, SP),
    backward_pass_(Net,D1,As,Zs,
		   [matrix:multiply(D1,matrix:transpose(A))|Nw],[D1|Nb]);
backward_pass_([],_,_,_,Nw,Nb) ->
    {Nw,Nb}.

cost_derivative(A, Y) ->
    matrix:subtract(A, Y).

%% Forward and store Z in a list
feed_forward_([#layer{weights=W,bias=B}|Net], A, As, Zs) ->
    Z = matrix:add(matrix:multiply(W, A), B),
    A1 = matrix:sigmoid(Z),
    feed_forward_(Net, A1, [A1|As], [Z|Zs]);
feed_forward_([], _A, As, Zs) ->
    {As, Zs}.

evaluate(Net, TestData) ->
    evaluate_(Net, TestData, 0).

evaluate_(Net, [{X,Y}|TestData], Sum) ->
    Y1 = feed(Net, X),
    Yi = matrix:element(1,1,matrix:argmax(Y1,0))-1,
    %% io:format("Y = ~w, Yi = ~w\n", [Y, Yi]),
    if Y =:= Yi -> evaluate_(Net, TestData, Sum+1);
       true -> evaluate_(Net, TestData, Sum)
    end;
evaluate_(_Net, [], Sum) ->
    Sum.

%% options
option(Key, Options) ->
    option(Key, Options, undefined).

option(Key, Options, Default) ->
    case maps:find(Key, Options) of
	{ok,Value} -> validate(Key, Value);
	%% check aliases
	error when Key =:= learning_rate -> option_(eta, Options, Default);
	error when Key =:= eta -> option_(learning_rate, Options, Default);
	error when Key =:= regularization ->  option_(lmbda, Options, Default);
	error when Key =:= lmbda -> option_(regularization, Options, Default);
	error -> validate(Key, Default)
    end.

option_(Key, Options, Default) ->
    validate(Key, maps:get(Key, Options, Default)).

validate(eta, Value) when is_number(Value), Value >= 0 -> Value;
validate(learning_rate, Value) when is_number(Value), Value >= 0 -> Value;
validate(lmbda, Value) when is_number(Value), Value >= 0 -> Value;
validate(regularization, Value) when is_number(Value), Value >= 0 -> Value;
validate(epochs, Value) when is_integer(Value), Value >= 0 -> Value;
validate(batch_size, Value) when is_integer(Value), Value >= 0 -> Value.
