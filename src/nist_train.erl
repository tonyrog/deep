%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    Train a neural network with the NIST data
%%% @end
%%% Created : 19 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(nist_train).

-compile(export_all).

%% read the NIST data in two parts
%% 50000 train data and 10000 validation data
%%

main() ->
    main(30,[30]).

main(Epochs,Hidden) ->
    {TrainingSet0,ValidationSet} = load2(5000,100),
    io:format("loaded 5000+100\n"),
    Net = deep_net:new(784,Hidden,10),
    TrainingSet = [ {X,nist:label_to_matrix(Y)} || {X,Y} <- TrainingSet0],
    deep_net:sgd(Net, TrainingSet, ValidationSet, Epochs, 10, 3.0).

main1() ->
    main1(30,[30]).

main1(Epochs,Hidden) ->
    {TrainingSet0,ValidationSet} = load2(50000,10000),
    io:format("loaded 50000+10000\n"),
    Net = deep_net:new(784,Hidden,10),
    TrainingSet = [ {X,nist:label_to_matrix(Y)} || {X,Y} <- TrainingSet0],
    deep_net:sgd(Net, TrainingSet, ValidationSet, Epochs, 10, 3.0).

load(N) ->
    {ok,Fd1} = nist:open_image_file(),
    {ok,Fd2} = nist:open_label_file(),
    R = load(Fd1,Fd2,N),
    nist:close_image_file(Fd1),
    nist:close_label_file(Fd2),
    R.

load2(N,M) ->
    {ok,Fd1} = nist:open_image_file(),
    {ok,Fd2} = nist:open_label_file(),
    Rn = load(Fd1,Fd2,N),
    Rm = load(Fd1,Fd2,M),
    nist:close_image_file(Fd1),
    nist:close_label_file(Fd2),
    {Rn,Rm}.

load(_Fd1,_Fd2,0) ->
    [];
load(Fd1,Fd2,I) ->
    {ok,Img} = nist:read_next_image(Fd1),
    {ok,<<L>>} = nist:read_next_label(Fd2),
    [{ nist:image_to_matrix(Img), L } | load(Fd1,Fd2,I-1)].
