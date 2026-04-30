module.exports = {
  apps: [
    {
      name: '评论写手服务',
      script: 'run.py',
      cwd: 'E:/评论写手',
      interpreter: 'E:/评论写手/commenter/Scripts/pythonw.exe',
      windowsHide: true,
      watch: false,
      autorestart: true,
      max_restarts: 20,
      max_memory_restart: '2G'
    }
  ]
};
