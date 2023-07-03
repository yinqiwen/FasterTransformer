/*
 * Copyright (c) 2022 TENCENT CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "file_utils.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace ft_ext {
int file_read_all(const std::string& path, std::string& content)
{
    FILE* fp;
    if ((fp = fopen(path.c_str(), "rb")) == NULL) {
        return -1;
    }
    content.clear();
    char buf[8192];
    while (true) {
        size_t len = fread(buf, 1, sizeof(buf), fp);
        if (len > 0) {
            content.append(buf, len);
        }
        else {
            break;
        }
    }
    fclose(fp);
    return 0;
}

bool is_file_exist(const std::string& path)
{
    if (::access(path.c_str(), F_OK) != -1) {
        return true;
    }
    else {
        return false;
    }
}

bool is_dir_exist(const std::string& path)
{
    struct stat buf;
    int         ret = stat(path.c_str(), &buf);
    if (0 == ret) {
        return S_ISDIR(buf.st_mode);
    }
    return false;
}

int list_subfiles(const std::string& path, std::vector<std::string>& fs)
{
    struct stat buf;
    int         ret = stat(path.c_str(), &buf);
    if (0 == ret) {
        if (S_ISDIR(buf.st_mode)) {
            DIR* dir = opendir(path.c_str());
            if (NULL != dir) {
                struct dirent* ptr;
                while ((ptr = readdir(dir)) != NULL) {
                    if (!strcmp(ptr->d_name, ".") || !strcmp(ptr->d_name, "..")) {
                        continue;
                    }
                    std::string file_path = path;
                    file_path.append("/").append(ptr->d_name);
                    memset(&buf, 0, sizeof(buf));
                    ret = stat(file_path.c_str(), &buf);
                    if (ret == 0) {
                        if (S_ISREG(buf.st_mode)) {
                            fs.push_back(ptr->d_name);
                        }
                    }
                }
                closedir(dir);
                return 0;
            }
        }
    }
    return -1;
}

int64_t file_size(const std::string& path)
{
    struct stat buf;
    int         ret      = stat(path.c_str(), &buf);
    int64_t     filesize = 0;
    if (0 == ret) {
        if (S_ISREG(buf.st_mode)) {
            return buf.st_size;
        }
        else if (S_ISDIR(buf.st_mode)) {
            DIR* dir = opendir(path.c_str());
            if (NULL != dir) {
                struct dirent* ptr;
                while ((ptr = readdir(dir)) != NULL) {
                    if (!strcmp(ptr->d_name, ".") || !strcmp(ptr->d_name, "..")) {
                        continue;
                    }
                    std::string file_path = path;
                    file_path.append("/").append(ptr->d_name);
                    filesize += file_size(file_path);
                }
                closedir(dir);
            }
        }
    }
    return filesize;
}

bool make_dir(const std::string& para_path)
{
    if (is_dir_exist(para_path)) {
        return true;
    }
    if (is_file_exist(para_path)) {
        // ERROR_LOG("Exist file '%s' is not a dir.", para_path.c_str());
        return false;
    }
    std::string path  = para_path;
    size_t      found = path.rfind("/");
    if (found == path.size() - 1) {
        path  = path.substr(0, path.size() - 1);
        found = path.rfind("/");
    }
    if (found != std::string::npos) {
        std::string base_dir = path.substr(0, found);
        if (make_dir(base_dir)) {
            // mode is 0755
            return mkdir(path.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0;
        }
    }
    else {
        return mkdir(path.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0;
    }
    return false;
}

bool make_file(const std::string& para_path)
{
    if (is_file_exist(para_path)) {
        return true;
    }
    if (is_dir_exist(para_path)) {
        // ERROR_LOG("Exist file '%s' is not a regular file.", para_path.c_str());
        return false;
    }
    std::string path  = para_path;
    size_t      found = path.rfind("/");
    if (found != std::string::npos) {
        std::string base_dir = path.substr(0, found);
        if (make_dir(base_dir)) {
            // mode is 0755
            return open(path.c_str(), O_CREAT, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0;
        }
    }
    else {
        return open(path.c_str(), O_CREAT, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0;
    }
    return false;
}

int file_write_content(const std::string& path, const void* content, size_t size)
{
    make_file(path);
    FILE* fp;
    if ((fp = fopen(path.c_str(), "wb")) == NULL) {
        return -1;
    }
    size_t writed = 0;
    if (size > 0) {
        writed = fwrite(content, 1, size, fp);
    }
    fclose(fp);
    return writed == size ? 0 : -1;
}

int file_write_content(const std::string& path, const std::string& content)
{
    return file_write_content(path, content.data(), content.size());
}

int file_read_content(const std::string& path, void* buf, uint32_t size)
{
    FILE* fp;
    if ((fp = fopen(path.c_str(), "rb")) == NULL) {
        return -1;
    }
    size_t len = fread(buf, size, 1, fp);
    fclose(fp);
    return len == 1 ? 0 : -1;
}

}  // namespace ft_ext