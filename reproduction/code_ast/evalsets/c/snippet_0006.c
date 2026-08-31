#ifndef PINOCCHIO_PLUGIN_STORE_H
#define PINOCCHIO_PLUGIN_STORE_H

#include <botan_all.h>

namespace Botan {
    class Private_Key;

    class RandomNumberGenerator;

    class BOTAN_PUBLIC_API(2, 0) Plugin_Certificate_Store : public Certificate_Store {
    public:
        explicit Plugin_Certificate_Store(const std::shared_ptr<SQL_Database> db,
                                          const std::string &passwd,
                                          RandomNumberGenerator &rng,
                                          const std::string &table_prefix = "");

        std::shared_ptr<const X509_Certificate>
        find_cert(const X509_DN &subject_dn, const std::vector<uint8_t> &key_id) const override;

        std::vector<std::shared_ptr<const X509_Certificate>> find_all_certs(
                const X509_DN &subject_dn, const std::vector<uint8_t> &key_id) const override;

        std::shared_ptr<const X509_Certificate>
        find_cert_by_pubkey_sha1(const std::vector<uint8_t> &key_hash) const override;

        std::shared_ptr<const X509_Certificate>
        find_cert_by_raw_subject_dn_sha256(const std::vector<uint8_t> &subject_hash) const override;

        std::vector<X509_DN> all_subjects() const override;

        bool insert_cert(const X509_Certificate &cert);

        bool remove_cert(const X509_Certificate &cert);

        std::shared_ptr<const Private_Key> find_key(const X509_Certificate &) const;

        std::vector<std::shared_ptr<const X509_Certificate>>
        find_certs_for_key(const Private_Key &key) const;

        bool insert_key(const X509_Certificate &cert, const Private_Key &key);

        void remove_key(const Private_Key &key);

        void revoke_cert(const X509_Certificate &, CRL_Code, const X509_Time &time = X509_Time());

        void affirm_cert(const X509_Certificate &);

        std::vector<X509_CRL> generate_crls() const;

        std::shared_ptr<const X509_CRL>
        find_crl_for(const X509_Certificate &issuer) const override;

    private:
        RandomNumberGenerator &m_rng;
        std::shared_ptr<SQL_Database> m_database;
        std::string m_prefix;
        std::string m_password;
        mutex_type m_mutex;
    };
}
#endif //PINOCCHIO_PLUGIN_STORE_H