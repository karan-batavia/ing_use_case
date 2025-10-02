# MongoDB Integration Documentation

## Overview

The ING Prompt Scrubber application now includes MongoDB integration to track user sessions, interactions, and application analytics.

## Features

### 📊 Session Tracking
- **Unique Session IDs**: Each user session gets a unique UUID
- **Session Duration**: Track start time, last activity, and end time
- **User Roles**: Differentiate between admin and regular users
- **Session Status**: Active/inactive session management

### 🔍 Interaction Logging
- **Text Input**: Log when users input text for scrubbing
- **File Uploads**: Track uploaded files (name, type, size)
- **Text Scrubbing**: Monitor scrubbing results and efficiency
- **File Downloads**: Log downloaded scrubbed files
- **Login/Logout**: Track authentication events

### 📈 Analytics & Insights
- **User Statistics**: Total sessions, interactions, and usage patterns
- **Session Statistics**: Detailed interaction breakdown per session
- **Admin Dashboard**: Real-time MongoDB connection status and stats
- **Performance Metrics**: Scrubbing efficiency and content reduction rates

## Database Schema

### Collections

#### `user_sessions`
```json
{
  "session_id": "uuid-string",
  "user_id": "user-identifier", 
  "user_role": "admin|user",
  "created_at": "timestamp",
  "last_activity": "timestamp",
  "is_active": true,
  "interactions_count": 0,
  "ended_at": "timestamp"
}
```

#### `user_interactions`
```json
{
  "interaction_id": "uuid-string",
  "session_id": "session-uuid",
  "user_id": "user-identifier",
  "action_type": "login|text_input|file_upload|text_scrubbing|file_download|logout",
  "timestamp": "timestamp",
  "details": {
    "filename": "optional-string",
    "file_type": "optional-string", 
    "file_size": "optional-number",
    "input_length": "optional-number",
    "output_length": "optional-number",
    "matches_found": "optional-number",
    "reduction_percentage": "optional-number"
  }
}
```

#### `users`
```json
{
  "user_id": "user-identifier",
  "user_role": "admin|user",
  "created_at": "timestamp",
  "last_login": "timestamp",
  "login_count": 0
}
```

## Configuration

### Environment Variables

- `MONGODB_URI`: MongoDB connection string (default: `mongodb://localhost:27017/`)
- `MONGODB_DB_NAME`: Database name (default: `ing_prompt_scrubber`)

### Docker Compose Setup

The application includes a complete MongoDB setup:

```yaml
services:
  mongodb:
    image: mongo:7.0
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password123
      - MONGO_INITDB_DATABASE=ing_prompt_scrubber
    volumes:
      - mongodb_data:/data/db
```

## Usage

### Starting with MongoDB

```bash
# Start the complete stack (app + MongoDB)
docker-compose up -d

# Check MongoDB connection
docker-compose logs mongodb

# Access MongoDB directly
docker exec -it ing-mongodb mongosh -u admin -p password123
```

### Graceful Degradation

The application works with or without MongoDB:

- **With MongoDB**: Full session tracking and analytics
- **Without MongoDB**: Basic functionality with console logging

### Admin Features

Administrators can view:
- 🟢 MongoDB connection status
- 📊 Real-time session statistics  
- 👥 User interaction counts
- 📈 Performance metrics

## API Reference

### MongoDB Service Methods

#### Session Management
- `create_session(user_id, user_role)`: Create new session
- `update_session_activity(session_id)`: Update last activity
- `end_session(session_id)`: Mark session as ended

#### Interaction Logging
- `log_interaction(session_id, user_id, action_type, details)`: Log general interaction
- `log_file_upload(session_id, user_id, filename, file_type, file_size)`: Log file upload
- `log_text_scrubbing(...)`: Log scrubbing results
- `log_file_download(...)`: Log file download
- `log_login(user_id, user_role)`: Log user authentication

#### Analytics
- `get_session_stats(session_id)`: Get session analytics
- `get_user_stats(user_id)`: Get user analytics

### Streamlit Integration

#### Convenience Functions
- `get_mongodb_service()`: Get MongoDB service instance
- `ensure_session_tracking()`: Initialize session tracking
- `log_app_interaction(action_type, details)`: Simple interaction logging

## Security Considerations

### Database Security
- Authentication enabled with username/password
- Network isolation through Docker networking
- Volume persistence for data durability

### Data Privacy
- User IDs are generated UUIDs (no personal data)
- File contents are not stored (only metadata)
- Session data automatically expires with inactivity

### Production Recommendations
- Use strong MongoDB credentials
- Enable MongoDB authentication and authorization
- Use TLS/SSL for MongoDB connections
- Implement data retention policies
- Regular database backups

## Monitoring & Maintenance

### Health Checks
- MongoDB: `mongosh --eval "db.adminCommand('ping')"`
- Connection status visible in admin dashboard

### Database Maintenance
```javascript
// Connect to MongoDB
use ing_prompt_scrubber

// View collection stats
db.user_sessions.stats()
db.user_interactions.stats()
db.users.stats()

// Clean old sessions (older than 30 days)
db.user_sessions.deleteMany({
  "created_at": {
    "$lt": new Date(Date.now() - 30*24*60*60*1000)
  }
})

// Create additional indexes
db.user_interactions.createIndex({"action_type": 1, "timestamp": -1})
```

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check if MongoDB container is running: `docker-compose ps`
   - Verify connection string in environment variables
   - Check MongoDB logs: `docker-compose logs mongodb`

2. **Performance Issues**
   - Monitor database indexes: Use MongoDB Compass
   - Check memory usage: `docker stats`
   - Implement data retention policies

3. **Data Not Appearing**
   - Verify session tracking is initialized
   - Check console logs for MongoDB errors
   - Ensure proper user authentication flow

### Debugging

Enable verbose logging:
```bash
# Set environment variable for detailed logs
export MONGODB_VERBOSE_LOGGING=true
docker-compose up
```

## Future Enhancements

### Planned Features
- 📊 Advanced analytics dashboard
- 📧 Email notifications for admin alerts
- 🔄 Data export/import functionality
- 📱 Mobile-responsive admin interface
- 🎯 Usage-based recommendations

### Potential Integrations
- Grafana dashboards for visualization
- Prometheus metrics collection
- Automated reporting and alerts
- Data warehouse integration