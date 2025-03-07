clear;
%load("weightsb.mat");
load("Data_Train.mat")
x=Data;
y=Result;

ynew=onehotencode(y,2,"ClassNames",[0;1]);

permatrix=transpose(randperm(4410));

Train_size=3087;

Input_data_train=x(permatrix(1:Train_size),:);
Input_data_test=x(permatrix((Train_size+1):end),:);

Output_data_train=ynew(permatrix(1:Train_size),:);
Output_data_test=ynew(permatrix((Train_size+1):end),:);

inputs=45;
outputs=2;

w_num1=30;
w_num2=outputs;


weights=cell(1,2);

weights{1,1}=normrnd(0,1,[inputs w_num1]);
weights{1,2}=normrnd(0,1,[w_num1 w_num2]);

biases=cell(1,2);

biases{1,1}=zeros(w_num1,1);
biases{1,2}=zeros(w_num2,1);

epoch=500;

plotdata1=zeros(1,epoch);
plotdata2=zeros(1,epoch);

for i=1:epoch
    track1=zeros(1,Train_size);
    track2=zeros(1,(4410-Train_size));

    for j=1:Train_size
        [~,b,c]=backprobagation(weights,biases,transpose(Input_data_train(j,:)),transpose(Output_data_train(j,:)),0.01);
        weights=b;
        biases=c;
    end

    for j=1:Train_size
        out=forwardcalc(weights,biases,transpose(Input_data_train(j,:)));
        n=find(out{end}==max(out{end}));
        m=find(Output_data_train(j,:)==1);
        if n==m
            track1(1,j)=1;
        end
    end

    for j=1:(4410-Train_size)
        out=forwardcalc(weights,biases,transpose(Input_data_test(j,:)));
        n=find(out{end}==max(out{end}));
        m=find(Output_data_test(j,:)==1);
        if n==m
            track2(1,j)=1;
        end
    end

    acc1=mean(track1);
    acc2=mean(track2);
    plotdata1(1,i)=acc1;
    plotdata2(1,i)=acc2;

    disp(['Run #',num2str(i),' Train Accuracy:',num2str(acc1),' Test Acurracy',num2str(acc2)]);
    

      

end

figure(1);

plot(1:epoch,plotdata1);

figure(2);

plot(1:epoch,plotdata2);


save weightsbb.mat x y weights biases Input_data_train Input_data_test Output_data_train Output_data_test;

function a=Sigmoid(input)
    a=1./(1+exp(-input));
end

function a=Relu(input)
    a=max(input,0);
end

function u=Softmax(input)
    temp=exp(input);
    u=temp/sum(temp);
end

function u=Softmaxd(output)
    temp=output*transpose(output);
    u=output.*eye(length(output))-temp;
end

function m=CE(output,desired)
    a=desired==1;
    m=-log(output(a));
end





function u=forwardcalc(weights,biases,input)  
    u=cell(1,2*length(weights)+1);
    u{1,1}=input;
    for i=1:length(weights)
        if i==length(weights)
            k=2*i;
            u{k} = transpose(weights{1,i}) * u{k-1} + biases{1,i};
            u{k+1}=Softmax(u{k});
        else
            k=2*i;
            u{k} = transpose(weights{1,i}) * u{k-1} + biases{1,i};
            u{k+1}=Relu(u{k});
        end
    end
end

function [lossbefore,weightsnew,biasesnew]=backprobagation(weights,biases,input,desired,learning_rate)
    output=forwardcalc(weights,biases,input);
    k=length(weights);

    lossbefore=CE(output{end},desired);


    delta=cell(1,k);
    dw=cell(1,k);
    dbiases=cell(1,k);

    delta{k}= output{2*k+1} - desired;
    dw{k}=output{2*k-1}*transpose(delta{k});
    dbiases{k}=delta{k};
    weightsnew{k}=weights{k}-dw{k}*learning_rate;
    biasesnew{k}=biases{k}-dbiases{k}*learning_rate;

   for i=k-1:-1:1
       delta{i}=(output{2*i+1}./output{2*i}).*weights{i+1}*delta{i+1};
       dw{i}=output{2*i-1}*transpose(delta{i});
       dbiases{i}=delta{i};
       weightsnew{i}=weights{i}-dw{i}*learning_rate;
       biasesnew{i}=biases{i}-dbiases{i}*learning_rate;
       
   end

   
end