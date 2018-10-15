%input P
%output: ignore s0
function [ B ] = Solve( P )
	M=size(P,1);
	status=zeros(M,M);
	obj=zeros(1,M);
	B=zeros(1,M-1);
	H=round(M/2);
	
	%calc dist
	dist=-bsxfun(@minus,bsxfun(@minus,2*P(:,1:M-1)'*P,sum(P(:,1:M-1).^2,1)'),sum(P.^2,1));
	
	minf=inf;minstatus=zeros(1,M);
	for i=1:M
		status(i,:)=find1(P,M,P(:,i),H,dist(:,i));
		obj(i)=calcObj(P,status(i,:),H);
	end
	
	[~,b]=min(obj);
	
	B=status(b,1:M-1)';
end

function [choose]=find1(P,M,p,H,d)

		choose=zeros(1,M);
		[~,ord]=sort(d);
		choose(ord(1:H-1))=1;
		choose(M)=1;
end




function [f]=calcObj(P,choose,H)
	%%compute geo center
	M=size(P,1);
	center=zeros(M,1);
	
	idx=find(choose==1);
	center=sum(P(:,idx),2)/H;
	
	f=sum(-bsxfun(@minus,bsxfun(@minus,2*P(:,idx)'*center,sum(P(:,idx).^2,1)'),sum(center.^2,1)));
end

