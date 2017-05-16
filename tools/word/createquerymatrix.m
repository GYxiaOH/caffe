
endfilename = sort(endfilename);
endquery = sparse(numel(endfilename),481998);
tindex = 1;
for i = 1:numel(endfilename)
    endquery(tindex,:) = img_Fea_Clickcount(endfilename(i),:);
    tindex = tindex + 1;
end

sumendquery = sum(endquery);
delquery = find(sumendquery==0);
delflag = 0 ;
for i = 1:numel(delquery)
    endquery(:,(delquery(i)-delflag)) = [];
    Dog_QQ_N((delquery(i) - delflag),:) = [];
    delflag = delflag + 1;
end