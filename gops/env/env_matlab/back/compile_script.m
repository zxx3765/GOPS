%一键编译脚本
clear all;
clc;
curren_path = pwd;
%获取当前文件夹下的文件
files = dir(curren_path);
%删除文件夹
for i = 1:length(files)
    if files(i).isdir == 1
        if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
            continue;
        end
        rmdir(files(i).name,'s');
    end
end
%删除后缀为.h、.json、.c、.cc、.py的文件和除了名字为model与env的.toml文件
for i = 1:length(files)
    if files(i).isdir == 0
        if strcmp(files(i).name(end-1:end),'.h') || strcmp(files(i).name(end-4:end),'.json') || strcmp(files(i).name(end-1:end),'.c') || strcmp(files(i).name(end-2:end),'.cc') || strcmp(files(i).name(end-2:end),'.py')
            delete(files(i).name);
        end
        if strcmp(files(i).name(end-4:end),'.toml')
            if ~strcmp(files(i).name(1:5),'model') && ~strcmp(files(i).name(1:3),'env')
                delete(files(i).name);
            end
        end
    end
end
%调用slxpy setup_config
slxpy.setup_config(curren_path);
%调用slxpy codegen
slxpy.codegen(curren_path);
% MATLAB script to call build_script.bat
batFileName = 'build_script.bat'; % Assuming the .bat file is in the current directory

% Execute the .bat file
[status, cmdout] = system(batFileName);

% Check if the command ran successfully
if status == 0
    disp('The batch file ran successfully!');
else
    disp('There was an error running the batch file.');
    disp(cmdout); % Display the command output for debugging
end

