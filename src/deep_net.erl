%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    deep_net
%%% @end
%%% Created : 14 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(deep_net).

-export([new/3]).
-export([feed/2]).
-export([sgd/4]).

-compile(export_all).

-define(DEFAULT_ACTIVATION, sigmoid).

-record(layer,
	{
	  inputs  :: integer(),         %% number of inputs
	  outputs  :: integer(),        %% number of outputs
	  weights :: matrix:matrix(),   %% weight matrix
	  bias    :: matrix:matrix(),   %% bias vector
	  locked = false :: boolean(),  %% update weights?
	  activation_fn :: fun((matrix:matrix()) -> matrix:matrix()),
	  gradient_fn :: fun((matrix:matrix()) -> matrix:matrix())
	}).

%% create a network with an input vector of size Ni 
%% then a list of sizes of the hidden layers, lastly the
%% output vector.
%%
new(IN, Hs, OUT) when is_integer(IN), IN > 0,
		      is_list(Hs) ->
    create__(IN, Hs++[OUT]).

create__(IN,[OUTL|Hs]) ->
    {Activation,OUT} =
	case OUTL of
	    {Act,Out} when is_atom(Act), is_integer(Out), Out > 0 ->
		{Act,Out};
	    Out when is_integer(Out), Out > 0 ->
		{?DEFAULT_ACTIVATION, Out}
	end,
    %% W = matrix:normal(OUT,IN,float32),  %% weights hidden layers
    Ws = [ matrix:to_list(matrix:scale(1/math:sqrt(IN),matrix:normal(1,IN,float32))) || _ <- lists:seq(1,OUT)],
    W  = matrix:from_list(lists:append(Ws),float32),
    B = matrix:normal(OUT,1,float32),   %% bias vector
    F = activation_function(Activation),
    G = gradient_function(Activation),
    L = #layer { inputs = IN,
		 outputs = OUT,
		 weights = W,
		 bias = B,
		 activation_fn = F,
		 gradient_fn = G
	       },
    [L | create__(OUT,Hs)];
create__(_IN,[]) ->
    [].

%%
%% Feed forward 
%%
feed(Net,IN) ->
    feed_(Net,IN).

feed_([#layer{weights=W,bias=B,activation_fn=F}|Net],A) ->
    Z = matrix:add(matrix:multiply(W,A), B),
    feed_(Net, F(Z));
feed_([], A) ->
    A.

sgd(Net,TrainingSet,ValidationSet,InputOptions)
      when is_list(TrainingSet),
	   is_list(ValidationSet),
	   is_map(InputOptions) ->
    Default = #{ eta => 3.0, 
		 lmbda => 0.0,
		 batch_size => 10,
		 epochs => 20,
		 k => 0 },
    Options = maps:merge(Default, InputOptions),
    Eta   = option(learning_rate, Options),
    Lmbda = option(regularization, Options),
    BatchSize = option(batch_size, Options),
    Epochs = option(epochs, Options),
    K = option(k, Options),
    Tn = length(TrainingSet),
    sgd_(Net,Tn,TrainingSet,ValidationSet,1,Epochs,BatchSize,Eta,Lmbda,K).

sgd_(Net,_Tn,_TrainingSet,_ValidationSet,E,Epochs,_BatchSize,_Eta,_Lmbda,_K)
  when E > Epochs -> Net;
sgd_(Net,Tn,TrainingSet,ValidationSet,E,Epochs,BatchSize,Eta,Lmbda,K) ->
    TrainingSet1 = deep_random:shuffle(TrainingSet),
    T0 = erlang:monotonic_time(),
    Net1 = sgd__(Net,Tn,TrainingSet1,BatchSize,Eta,Lmbda,K),
    T1 = erlang:monotonic_time(),
    R = evaluate(Net1,ValidationSet),
    T2 = erlang:monotonic_time(),
    M = length(ValidationSet),
    Time1 = erlang:convert_time_unit(T1 - T0, native, microsecond),
    Time2 = erlang:convert_time_unit(T2 - T1, native, microsecond),
    io:format("epoch ~w/~w : learn=~.2f%, train=~.2fs, eval=~.2fs\n", 
	      [E,Epochs,(R/M)*100,Time1/1000000,Time2/1000000]),
    sgd_(Net1,Tn,TrainingSet,ValidationSet,E+1,Epochs,BatchSize,Eta,Lmbda,K).

%% loop over batches and train the net
sgd__(Net, _Tn, [], _BatchSize, _Eta, _Lmbda, _K) ->
    Net;
sgd__(Net, Tn, TrainingSet, BatchSize, Eta, Lmbda, K) ->
    Batch = lists:sublist(TrainingSet, BatchSize),
    Bn = length(Batch),
    Net1 = learn1(Net, Batch, Eta, Bn, Lmbda, Tn, K),
    %% Net1 = learn(Net, Batch, Eta, Bn, Lmbda, Tn),
    TrainingSet1 = lists:nthtail(Bn, TrainingSet),
    sgd__(Net1, Tn, TrainingSet1, BatchSize, Eta, Lmbda, K).

%%
%% Run learning over a batch of {X,Y} pairs
%% That mean update the weights after each batch number of
%% paris
%%
learn(Net, [{X,Y}|Batch], Eta, Bn, Lmbda, Tn) ->
    {Ws,Bs} = backprop(Net, X, Y),
    learn(Net, Batch, Eta, Bn, Lmbda, Tn, Ws, Bs).

learn(Net, [{X,Y}|Batch], Eta, Bn, Lmbda, Tn, Ws, Bs) ->
    {WsB,BsB} = backprop(Net, X, Y),
    Ws1 = lists:zipwith(fun matrix:add/2, Ws, WsB),
    Bs1 = lists:zipwith(fun matrix:add/2, Bs, BsB),
    learn(Net, Batch, Eta, Bn, Lmbda, Tn, Ws1, Bs1);
learn(Net, [], Eta, Bn, Lmbda, Tn, Ws, Bs) ->
    learn_net(Net, Ws, Bs, Eta, Bn, Lmbda, Tn).

%% Bn = mini batch size
%% Tn = training set size
learn_net([L=#layer{locked=true}|Net],[_NW|Ws],[_NB|Bs],Eta,Bn,Lmbda,Tn) ->
    [L | learn_net(Net, Ws, Bs, Eta, Bn, Lmbda, Tn)];
learn_net([L=#layer{weights=W,bias=B}|Net],[NW|Ws],[NB|Bs],Eta,Bn,Lmbda,Tn) ->
    W1 = matrix:subtract(matrix:scale((1-Eta*Lmbda/Tn),W),
			 matrix:scale(Eta/Bn,NW)),
    B1 = matrix:subtract(B, matrix:scale(Eta/Bn, NB)),
    [L#layer{weights=W1,bias=B1} | learn_net(Net,Ws,Bs,Eta,Bn,Lmbda,Tn)];
learn_net([], [], [], _Eta, _Bn, _Lmbda, _Tn) ->
    [].

%%
%% back propagation algorithm,
%% adjust weights and biases to match output Y on input X
%%
backprop(Net=[#layer{gradient_fn=G}|Net1], X, Y) ->
    {[Out,A|As],[Z|Zs]} = feed_forward_(Net, X, [X], []),
    Delta = matrix:times(cost_derivative(Out, Y), G(Z)),
    %% select ktop from cost_derivative => Delta
    W1 = matrix:multiply(Delta,matrix:transpose(A)),
    NetR = lists:reverse(Net1),
    backward_pass_(NetR,Delta,As,Zs,[W1],[Delta]).

backward_pass_([#layer{weights=W,bias=_B,gradient_fn=G}|Net],
	       Delta0,[A|As],[Z|Zs],Nw,Nb) ->
    %% Transform Delta into column order (for speed)
    DeltaC   = matrix:transpose(matrix:transpose_data(Delta0)),
    Wt = matrix:transpose(W),
    WtD  = matrix:multiply(Wt, DeltaC),
    Delta = matrix:times(WtD, G(Z)),
    W1 = matrix:multiply(Delta,matrix:transpose(A)),
    backward_pass_(Net,Delta,As,Zs,[W1|Nw],[Delta|Nb]);
backward_pass_([],_,_,_,Nw,Nb) ->
    {Nw,Nb}.

%%
%% learn1 is feed and backprop in one go.
%% weights are updated on recursive return
%% (batch is not really used in this case)
%%
learn1(Net, [{X,Y}|Batch], Eta, Bn, Lmbda, Tn, K) ->
    {_, Net1} = learn1_(1, Net, X, Y, Eta, Lmbda, K),
    learn1(Net1, Batch, Eta, Bn, Lmbda, Tn, K);
learn1(Net, [], _Eta, _Bb, _Lmbda, _Tn, _K) ->
    Net.
    
learn1_(_I,[L=#layer{weights=W,bias=B,activation_fn=F,gradient_fn=G}],
	A, Y, Eta, Lmbda, K) ->
    Z = matrix:add(matrix:multiply(W, A), B),
    Out = F(Z),
    Grad = G(Z),
    Ki = matrix:topk(Grad,K),
    Delta = matrix:ktimes(Ki,cost_derivative(Out, Y), Grad),
    W1 = matrix:kmultiply(Ki,Delta,matrix:transpose(A)),
    NW1 = matrix:subtract(matrix:scale((1-Eta*Lmbda),W),matrix:scale(Eta,W1)),
    NB1 = matrix:subtract(B, matrix:scale(Eta, Delta)),
    {Delta,[L#layer{weights=NW1,bias=NB1}]};
learn1_(I, [L=#layer{weights=W,bias=B,activation_fn=F,gradient_fn=G}|
	    Net=[#layer{weights=Wn}|_]],
	A, Y, Eta, Lmbda,K) ->
    Z = matrix:add(matrix:multiply(W,A),B),
    A1 = F(Z),
    {Delta0,Net1} = learn1_(I+1,Net,A1,Y,Eta,Lmbda,K),
    DeltaC = matrix:transpose(matrix:transpose_data(Delta0)),
    Wt = matrix:transpose(Wn), %% Note Wn!!!
    WtD = matrix:multiply(Wt,DeltaC),
    Grad = G(Z),
    Ki = matrix:topk(Grad,K),
    Delta = matrix:ktimes(Ki,WtD, Grad),
    W1 = matrix:kmultiply(Ki,Delta,matrix:transpose(A)),
    NW1 = matrix:subtract(matrix:scale((1-Eta*Lmbda),W),matrix:scale(Eta,W1)),
    NB1 = matrix:subtract(B, matrix:scale(Eta, Delta)),
    {Delta,[L#layer{weights=NW1,bias=NB1}|Net1]}.


cost_derivative(A, Y) ->
    matrix:subtract(A, Y).

%% Forward and store Z in a list
feed_forward_([#layer{weights=W,bias=B,activation_fn=F}|Net], A, As, Zs) ->
    Z = matrix:add(matrix:multiply(W, A), B),
    A1 = F(Z),
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

validate(k, Value) when is_integer(Value), Value >= 0 -> Value;
validate(eta, Value) when is_number(Value), Value >= 0 -> Value;
validate(learning_rate, Value) when is_number(Value), Value >= 0 -> Value;
validate(lmbda, Value) when is_number(Value), Value >= 0 -> Value;
validate(regularization, Value) when is_number(Value), Value >= 0 -> Value;
validate(epochs, Value) when is_integer(Value), Value >= 0 -> Value;
validate(batch_size, Value) when is_integer(Value), Value >= 0 -> Value.

activation_function(sigmoid) -> fun matrix:sigmoid/1;
activation_function(tanh) ->  fun matrix:tanh/1;
activation_function(relu) -> fun matrix:relu/1;
activation_function(leaky_relu) -> fun matrix:leaky_relu/1;
activation_function(linear) -> fun matrix:linear/1;
activation_function(softplus) -> fun matrix:softplus/1.

gradient_function(sigmoid) -> fun matrix:sigmoid_prime/1;
gradient_function(tanh) ->  fun matrix:tanh_prime/1;
gradient_function(relu) -> fun matrix:relu_prime/1;
gradient_function(leaky_relu) -> fun matrix:leaky_relu_prime/1;
gradient_function(linear) -> fun matrix:linear_prime/1;
gradient_function(softplus) -> fun matrix:softplus_prime/1.
    
