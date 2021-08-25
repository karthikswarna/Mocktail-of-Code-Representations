import os
import json
import requests
import subprocess

def get_top_repos(minPage = 1, maxpage = 20):
    repo_details = []
    for i in range(minPage, maxpage + 1):
        try:
            public_repos = requests.get('https://api.github.com/search/repositories?page=' + str(i) + '&q=language:C&sort=stars&order=desc').json()['items']
        except:
            print("Exited at {}. Use it as minPage next time".format(i))
            return repo_details

        for repo in public_repos:
            repo_name = repo['name']
            repo_link = repo['html_url']
            repo_stars = repo['stargazers_count']
            repo_forks = repo['forks_count']
            repo_size = repo['size']
            # print(repo_name, repo_link, repo_stars)
            repo_details.append({ 'name': repo_name,
                                    'link': repo_link,
                                    'stars': repo_stars,
                                    'forks': repo_forks,
                                    'size': repo_size})

    return repo_details


def clone_repos(filename="repos.json", repoStartIndex=0, repoEndIndex=9):
    existingRepos = [os.path.join('CppDataset', folder) for folder in os.listdir('CppDataset') if os.path.isdir(os.path.join('CppDataset', folder))]

    with open(filename, "r") as f:
        repo_list = json.load(f)
        repoCount = 0

        for repo_name, repo_link in repo_list.items():
            if repoCount < repoStartIndex:
                repoCount += 1
                continue
            if repoCount > repoEndIndex:
                break

            in_path = os.path.join("CppDataset", repo_name)
            if in_path not in existingRepos:
                # p = subprocess.Popen("git clone --depth=1 --progress -v " + repo_link + ' ' + in_path, stdout=subprocess.PIPE)
                p = subprocess.Popen(["git", "clone", "--depth=1", "--progress", "-v", repo_link, in_path], stdout=subprocess.PIPE)
                p.wait()

            repoCount += 1


if __name__ == '__main__':
    # Stars C:
    # print(json.dumps(get_top_repos(35, 200), indent=4))

    # Forks C:
    # pretty(get_top_repos(0, 200))
    # print(json.dumps(get_top_repos(34, 200), indent=4))

    # clone_repos("repos.json", repoStartIndex=0, repoEndIndex=998)       # 0 Indexing