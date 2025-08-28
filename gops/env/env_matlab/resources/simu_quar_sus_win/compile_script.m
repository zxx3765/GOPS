%一键编译脚本

%% 请注意 需要进入该目录下，启用conda环境 通过终端命令行打开matlab软件
clear;
clc;
curren_path = pwd;
% %获取当前文件夹下的文件
% files = dir(curren_path);
% %删除文件夹
% for i = 1:length(files)
%     if files(i).isdir == 1
%         if strcmp(files(i).name,'.') || strcmp(files(i).name,'..') || strcmp(files(i).name,'sim_Data')
%             continue;
%         end
%         rmdir(files(i).name,'s');
%     end
% end
%删除后缀为.h、.json、.c、.cc、.py、txt的文件和除了名字为model与env的.toml文件
%保留setup_fixbug.py文件
% for i = 1:length(files)
%     if files(i).isdir == 0
%         % 检查是否为setup_fixbug.py文件，如果是则跳过
%         if strcmp(files(i).name, 'setup_fixbug.py')
%             continue;
%         end
%         if strcmp(files(i).name(end-1:end),'.h') || strcmp(files(i).name(end-4:end),'.json') || strcmp(files(i).name(end-1:end),'.c') || strcmp(files(i).name(end-2:end),'.cc') || strcmp(files(i).name(end-2:end),'.py') || strcmp(files(i).name(end-3:end),'.txt')
%             delete(files(i).name);
%         end
%         if strcmp(files(i).name,'model.toml') || strcmp(files(i).name,'env.toml')
%             continue;
%         end
%     end
% end
%调用slxpy setup_config
slxpy.setup_config(curren_path);
%调用slxpy codegen
slxpy.codegen(curren_path);
%% 目前有bug 先手动执行bat文件
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
%% 

%拷贝编译文件到GOPS
files = dir([curren_path,'\build']);
%将build文件夹下的名为lib.*文件夹下所有文件和文件夹移动到当前文件夹下
for i = 1:length(files)
    if files(i).isdir == 1
        if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
            continue;
        end
        %判断文件夹名字是否为lib.*
        if strcmp(files(i).name(1:4),'lib.')
            %获取lib.*文件夹下的文件
            lib_files = dir([curren_path,'\build\',files(i).name]);
            %将lib.*文件夹下的文件和文件夹移动到当前文件夹下
            for j = 1:length(lib_files)
                if lib_files(j).isdir == 1
                    if strcmp(lib_files(j).name,'.') || strcmp(lib_files(j).name,'..')
                        continue;
                    end
                    copyfile([curren_path,'\build\',files(i).name,'\',lib_files(j).name],[curren_path,'\',lib_files(j).name]);
                else
                    copyfile([curren_path,'\build\',files(i).name,'\',lib_files(j).name],[curren_path,'\',lib_files(j).name]);
                end
            end
        end
    end
end

%% 拷贝文件到GOPS环境目录
% 获取当前目录名
[~, current_dir_name, ~] = fileparts(curren_path);
% 设置目标路径，使用当前目录名
target_path = ['D:\Project\GOPS\gops\env\env_matlab\resources\', current_dir_name];

% 检查目标目录是否存在，不存在则创建
if ~exist(target_path, 'dir')
    % mkdir(target_path);
    disp(['Created target directory: ', target_path]);
end

files = dir([curren_path,'\build']);
%将build文件夹下的名为lib.*文件夹下所有文件和文件夹移动到当前文件夹下
for i = 1:length(files)
    if files(i).isdir == 1
        if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
            continue;
        end
        %判断文件夹名字是否为lib.*
        if strcmp(files(i).name(1:4),'lib.')
            %获取lib.*文件夹下的文件
            lib_files = dir([curren_path,'\build\',files(i).name]);
            %将lib.*文件夹下的文件和文件夹移动到当前文件夹下
            for j = 1:length(lib_files)
                if lib_files(j).isdir == 1
                    if strcmp(lib_files(j).name,'.') || strcmp(lib_files(j).name,'..')
                        continue;
                    end
                    copyfile([curren_path,'\build\',files(i).name,'\',lib_files(j).name],[target_path,'\',lib_files(j).name]);
                else
                    copyfile([curren_path,'\build\',files(i).name,'\',lib_files(j).name],[target_path,'\',lib_files(j).name]);
                end
            end
        end
    end
end

disp('File copying completed!');
%% 

