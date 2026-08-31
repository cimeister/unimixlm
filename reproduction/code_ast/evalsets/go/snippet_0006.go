package database

import (
	"fmt"

	log "github.com/sirupsen/logrus"
	"gorm.io/driver/postgres" // using postgres sql
	"gorm.io/gorm"

	"github.com/aeekayy/go-api-base/pkg/models"
)

// SetupDatabase sets up a database connection with DBConfig
func SetupDatabase(config *DBConfig) (*gorm.DB, error) {
	// To get the value from the config file using key
	// viper package read .env
	viperUser := config.Username
	viperPassword := config.Password
	viperDB := config.DBName
	viperHost := config.Host
	viperPort := config.Port

	postgresConname := fmt.Sprintf("host=%v port=%v user=%v dbname=%v password=%v sslmode=disable", viperHost, viperPort, viperUser, viperDB, viperPassword)

	log.Infof("conname is\t\t%s", postgresConname)

	db, err := gorm.Open(postgres.Open(postgresConname), &gorm.Config{})
	if err != nil {
		log.Error("Failed to connect to database!")
		return nil, err
	}

	log.Info("Connected to the database")

	log.Infof("Auto-migrate is set to %+v", config.AutoMigrate)

	if config.AutoMigrate {
		log.Info("Moving Migration")
	}

	return db, nil
}

// MigrateDatabase migrates the database
// select district models from the models package
func MigrateDatabase(sqlDB *gorm.DB, config *DBConfig) error {
	log.Info("Migrating the database")
	var err error

	if sqlDB == nil {
		// To get the value from the config file using key
		// viper package read .env
		viperUser := config.Username
		viperPassword := config.Password
		viperDB := config.DBName
		viperHost := config.Host
		viperPort := config.Port

		postgresConname := fmt.Sprintf("host=%v port=%v user=%v dbname=%v password=%v sslmode=disable", viperHost, viperPort, viperUser, viperDB, viperPassword)

		log.Infof("conname is\t\t%s", postgresConname)

		sqlDB, err = gorm.Open(postgres.Open(postgresConname), &gorm.Config{})
		if err != nil {
			log.Error("Error detected with the migration")
			return err
		}
	}

	log.Info("Migrating users")
	sqlDB.AutoMigrate(&models.User{})
	log.Info("Migrating events")
	sqlDB.AutoMigrate(&models.Event{})
	log.Info("Database migration is complete")

	return nil
}