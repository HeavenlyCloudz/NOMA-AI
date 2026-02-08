**NOMA AI**


# Make startup script executable
chmod +x /home/havil/noma_ai/start_noma.sh

# Install service
sudo cp noma_ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable noma_ai.service
sudo systemctl start noma_ai.service

# Check status
sudo systemctl status noma_ai.service

# View logs
sudo journalctl -u noma_ai.service -f
