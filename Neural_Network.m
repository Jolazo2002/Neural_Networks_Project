clear;
%load("weightsb.mat");
load("Data_input_output.mat")
x=(data-mean(data)) ./std(data);
y=Desired;

ynew=onehotencode(y,2,"ClassNames",[-1;0;1]);

permatrix=transpose(randperm(4410));

Train_size=3087;

Input_data_train=x(permatrix(1:Train_size),:);
Input_data_test=x(permatrix((Train_size+1):end),:);

Output_data_train=ynew(permatrix(1:Train_size),:);
Output_data_test=ynew(permatrix((Train_size+1):end),:);

inputs=85;
outputs=3;

w_num1=60;
w_num2=outputs;


weights=cell(1,2);

weights{1,1}=normrnd(0,1,[inputs w_num1]);
weights{1,2}=normrnd(0,1,[w_num1 w_num2]);

biases=cell(1,2);

biases{1,1}=zeros(w_num1,1);
biases{1,2}=zeros(w_num2,1);


plotdata=zeros(1,100);

for i=1:1000
    loss1=zeros(1,4);
    loss2=zeros(1,4);

    for j=1:Train_size
        [loss1(j),b,c]=backprobagation(weights,biases,transpose(Input_data_train(j,:)),transpose(Output_data_train(j,:)),0.01);
        weights=b;
        biases=c;
        output=forwardcalc(weights,biases,transpose(Input_data_train(j,:)));
        loss2(j)=(1/2)*sum((output{end}-transpose(Output_data_train(j,:))).^2);
    end

    n=mean(loss1);
    m=mean(loss2);
    plotdata(i)=n;

    disp(['Run #',num2str(i),' Error Before:',num2str(n),' Error After:',num2str(n)]);
    

   %{ 
    [a,b,c]=backprobagation(weights,biases,transpose(x(1,:)),y(1),0.01);
    weights=b;
    biases=c;
    output=forwardcalc(weights,biases,transpose(x(1,:)));
    lossafter=(1/2)*sum((output{end}-y(1)).^2);
    disp(['Run #',num2str(i),' Error Before:',num2str(a),' Error After:',num2str(lossafter)]); 
   %}

end

plot(1:1000,plotdata);


track1=zeros(1,Train_size);
track2=zeros(1,(4410-Train_size));

for i=1:Train_size
    out=forwardcalc(weights,biases,transpose(Input_data_train(i,:)));
    n=find(out{end}==max(out{end}));
    m=find(Output_data_train(i,:)==1);
    if n==m
        track1(1,i)=1;
    end
end

for i=1:(4410-Train_size)
    out=forwardcalc(weights,biases,transpose(Input_data_test(i,:)));
    n=find(out{end}==max(out{end}));
    m=find(Output_data_test(i,:)==1);
    if n==m
        track2(1,i)=1;
    end
end

acc1=mean(track1);
acc2=mean(track2);

disp([num2str(acc1),' ',num2str(acc2)]);

save weights.mat weights biases;

function a=Sigmoid(input)
    a=1./(1+exp(-input));
end



function u=forwardcalc(weights,biases,input)  
    u=cell(1,2*length(weights)+1);
    u{1,1}=input;
    for i=1:length(weights)
        k=2*i;
        u{k} = transpose(weights{1,i}) * u{k-1} + biases{1,i};
        u{k+1}=Sigmoid(u{k});
    end
end

function [lossbefore,weightsnew,biasesnew]=backprobagation(weights,biases,input,desired,learning_rate)
    output=forwardcalc(weights,biases,input);
    k=length(weights);

    lossbefore=(1/2)*sum((output{end}-desired).^2);


    delta=cell(1,k);
    dw=cell(1,k);
    dbiases=cell(1,k);

    delta{k}=(output{2*k+1}-desired).*(output{2*k+1}.*(1-output{2*k+1}));
    dw{k}=output{2*k-1}*transpose(delta{k});
    dbiases{k}=delta{k};
    weightsnew{k}=weights{k}-dw{k}*learning_rate;
    biasesnew{k}=biases{k}-dbiases{k}*learning_rate;

   for i=k-1:-1:1
       delta{i}=(weights{i+1}*delta{i+1}).*(output{2*i+1}.*(1-output{2*i+1}));
       dw{i}=output{2*i-1}*transpose(delta{i});
       dbiases{i}=delta{i};
       weightsnew{i}=weights{i}-dw{i}*learning_rate;
       biasesnew{i}=biases{i}-dbiases{i}*learning_rate;
       
   end

   
end